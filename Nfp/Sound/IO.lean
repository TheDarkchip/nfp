-- SPDX-License-Identifier: AGPL-3.0-or-later

import Std
import Nfp.Sound.BinaryPure
import Nfp.Sound.Cert
import Nfp.Sound.HeadCert
import Nfp.Sound.ModelHeader
import Nfp.Sound.TextPure
import Nfp.Untrusted.SoundBinary
import Nfp.Untrusted.SoundCompute

namespace Nfp.Sound

open IO

/-!
# SOUND IO wrappers (trusted verification only)

This module is intentionally thin: it delegates witness generation to
`Nfp.Untrusted.SoundCompute` and **verifies** returned certificates locally.
-/

private def readTextModelHeader (path : System.FilePath) :
    IO (Except String TextHeader) := do
  let contents ← IO.FS.readFile path
  let lines : Array String := (contents.splitOn "\n").toArray
  return Nfp.Sound.parseTextHeader lines

private def readBinaryModelHeader (path : System.FilePath) :
    IO (Except String Nfp.Sound.BinaryHeader) := do
  IO.FS.withFile path IO.FS.Mode.read fun h => do
    match ← Nfp.Untrusted.SoundBinary.readBinaryHeader h with
    | .error e => return .error e
    | .ok hdr => return .ok hdr

private def readModelHeader (path : System.FilePath) :
    IO (Except String (Rat × GeluDerivTarget)) := do
  let firstLine ←
    IO.FS.withFile path IO.FS.Mode.read fun h => h.getLine
  if firstLine.trim = "NFP_BINARY_V1" then
    match ← readBinaryModelHeader path with
    | .error e => return .error e
    | .ok hdr => return .ok (hdr.eps, hdr.geluDerivTarget)
  else
    match ← readTextModelHeader path with
    | .error e => return .error e
    | .ok hdr => return .ok (hdr.eps, hdr.geluDerivTarget)

private def readModelEps (path : System.FilePath) : IO (Except String Rat) := do
  match ← readModelHeader path with
  | .error e => return .error e
  | .ok (eps, _) => return .ok eps

private def recomputeModelWeightBoundsBinary
    (path : System.FilePath) : IO (Except String ModelWeightBounds) := do
  let h ← IO.FS.Handle.mk path IO.FS.Mode.read
  match ← Nfp.Untrusted.SoundBinary.readBinaryHeader h with
  | .error e => return .error e
  | .ok hdr =>
      let scalePow10 := defaultBinaryScalePow10
      match ← Nfp.Untrusted.SoundBinary.skipI32Array h hdr.seqLen with
      | .error e => return .error e
      | .ok _ => pure ()
      match ← Nfp.Untrusted.SoundBinary.skipF64Array h (hdr.seqLen * hdr.modelDim) with
      | .error e => return .error e
      | .ok _ => pure ()
      let mut valuePairsLayers : Array (Array (Int × Int)) := Array.mkEmpty hdr.numLayers
      let mut qkPairsLayers : Array (Array (Int × Int)) := Array.mkEmpty hdr.numLayers
      let mut mlpWinBound : Array Rat := Array.mkEmpty hdr.numLayers
      let mut mlpWoutBound : Array Rat := Array.mkEmpty hdr.numLayers
      let mut ln1MaxAbsGamma : Array Rat := Array.mkEmpty hdr.numLayers
      let mut ln1MaxAbsBeta : Array Rat := Array.mkEmpty hdr.numLayers
      let mut ln2MaxAbsGamma : Array Rat := Array.mkEmpty hdr.numLayers
      for _l in [:hdr.numLayers] do
        let mut valuePairs : Array (Int × Int) := Array.mkEmpty hdr.numHeads
        let mut qkPairs : Array (Int × Int) := Array.mkEmpty hdr.numHeads
        for _h in [:hdr.numHeads] do
          let wqScaledE ←
            Nfp.Untrusted.SoundBinary.readMatrixNormInfScaled
              h hdr.modelDim hdr.headDim scalePow10
          let wqScaled ←
            match wqScaledE with
            | .error e => return .error e
            | .ok v => pure v
          match ← Nfp.Untrusted.SoundBinary.skipF64Array h hdr.headDim with
          | .error e => return .error e
          | .ok _ => pure ()
          let wkScaledE ←
            Nfp.Untrusted.SoundBinary.readMatrixNormInfScaled
              h hdr.modelDim hdr.headDim scalePow10
          let wkScaled ←
            match wkScaledE with
            | .error e => return .error e
            | .ok v => pure v
          match ← Nfp.Untrusted.SoundBinary.skipF64Array h hdr.headDim with
          | .error e => return .error e
          | .ok _ => pure ()
          let nvScaledE ←
            Nfp.Untrusted.SoundBinary.readMatrixNormInfScaled
              h hdr.modelDim hdr.headDim scalePow10
          let nvScaled ←
            match nvScaledE with
            | .error e => return .error e
            | .ok v => pure v
          match ← Nfp.Untrusted.SoundBinary.skipF64Array h hdr.headDim with
          | .error e => return .error e
          | .ok _ => pure ()
          let noScaledE ←
            Nfp.Untrusted.SoundBinary.readMatrixNormInfScaled
              h hdr.headDim hdr.modelDim scalePow10
          let noScaled ←
            match noScaledE with
            | .error e => return .error e
            | .ok v => pure v
          qkPairs := qkPairs.push (wqScaled, wkScaled)
          valuePairs := valuePairs.push (nvScaled, noScaled)
        match ← Nfp.Untrusted.SoundBinary.skipF64Array h hdr.modelDim with
        | .error e => return .error e
        | .ok _ => pure ()
        let nWinScaledE ←
          Nfp.Untrusted.SoundBinary.readMatrixNormInfScaled
            h hdr.modelDim hdr.hiddenDim scalePow10
        let nWinScaled ←
          match nWinScaledE with
          | .error e => return .error e
          | .ok v => pure v
        match ← Nfp.Untrusted.SoundBinary.skipF64Array h hdr.hiddenDim with
        | .error e => return .error e
        | .ok _ => pure ()
        let nWoutScaledE ←
          Nfp.Untrusted.SoundBinary.readMatrixNormInfScaled
            h hdr.hiddenDim hdr.modelDim scalePow10
        let nWoutScaled ←
          match nWoutScaledE with
          | .error e => return .error e
          | .ok v => pure v
        match ← Nfp.Untrusted.SoundBinary.skipF64Array h hdr.modelDim with
        | .error e => return .error e
        | .ok _ => pure ()
        let ln1GammaScaledE ←
          Nfp.Untrusted.SoundBinary.readVectorMaxAbsScaled h hdr.modelDim scalePow10
        let ln1GammaScaled ←
          match ln1GammaScaledE with
          | .error e => return .error e
          | .ok v => pure v
        let ln1BetaScaledE ←
          Nfp.Untrusted.SoundBinary.readVectorMaxAbsScaled h hdr.modelDim scalePow10
        let ln1BetaScaled ←
          match ln1BetaScaledE with
          | .error e => return .error e
          | .ok v => pure v
        let ln2GammaScaledE ←
          Nfp.Untrusted.SoundBinary.readVectorMaxAbsScaled h hdr.modelDim scalePow10
        let ln2GammaScaled ←
          match ln2GammaScaledE with
          | .error e => return .error e
          | .ok v => pure v
        let ln2BetaScaledE ←
          Nfp.Untrusted.SoundBinary.readVectorMaxAbsScaled h hdr.modelDim scalePow10
        match ln2BetaScaledE with
        | .error e => return .error e
        | .ok _ => pure ()
        let nWin := ratOfScaledInt scalePow10 nWinScaled
        let nWout := ratOfScaledInt scalePow10 nWoutScaled
        let ln1Gamma := ratOfScaledInt scalePow10 ln1GammaScaled
        let ln1Beta := ratOfScaledInt scalePow10 ln1BetaScaled
        let ln2Gamma := ratOfScaledInt scalePow10 ln2GammaScaled
        mlpWinBound := mlpWinBound.push nWin
        mlpWoutBound := mlpWoutBound.push nWout
        ln1MaxAbsGamma := ln1MaxAbsGamma.push ln1Gamma
        ln1MaxAbsBeta := ln1MaxAbsBeta.push ln1Beta
        ln2MaxAbsGamma := ln2MaxAbsGamma.push ln2Gamma
        valuePairsLayers := valuePairsLayers.push valuePairs
        qkPairsLayers := qkPairsLayers.push qkPairs
      match ← Nfp.Untrusted.SoundBinary.skipF64Array h hdr.modelDim with
      | .error e => return .error e
      | .ok _ => pure ()
      match ← Nfp.Untrusted.SoundBinary.skipF64Array h hdr.modelDim with
      | .error e => return .error e
      | .ok _ => pure ()
      match ← Nfp.Untrusted.SoundBinary.skipF64Array h (hdr.modelDim * hdr.vocabSize) with
      | .error e => return .error e
      | .ok _ => pure ()
      match attnWeightBoundsArraysFromScaledPairs scalePow10 valuePairsLayers qkPairsLayers with
      | .error e => return .error e
      | .ok (coeffs, wqMaxs, wkMaxs) =>
          return .ok {
            attnValueCoeff := coeffs
            wqOpBoundMax := wqMaxs
            wkOpBoundMax := wkMaxs
            mlpWinBound := mlpWinBound
            mlpWoutBound := mlpWoutBound
            ln1MaxAbsGamma := ln1MaxAbsGamma
            ln1MaxAbsBeta := ln1MaxAbsBeta
            ln2MaxAbsGamma := ln2MaxAbsGamma
          }

private def recomputeModelWeightBoundsText
    (path : System.FilePath) : IO (Except String ModelWeightBounds) := do
  let contents ← IO.FS.readFile path
  let lines : Array String := (contents.splitOn "\n").toArray
  return modelWeightBoundsFromTextLines lines

private def recomputeModelWeightBounds
    (path : System.FilePath) : IO (Except String ModelWeightBounds) := do
  let firstLine ←
    IO.FS.withFile path IO.FS.Mode.read fun h => h.getLine
  if firstLine.trim = "NFP_BINARY_V1" then
    recomputeModelWeightBoundsBinary path
  else
    recomputeModelWeightBoundsText path

/-- Compute weight-only per-head contribution bounds from a binary `.nfpt`. -/
def certifyHeadBoundsBinary
    (path : System.FilePath)
    (scalePow10 : Nat := 9) :
    IO (Except String (Array HeadContributionCert)) := do
  match ← Nfp.Untrusted.SoundCompute.certifyHeadBoundsBinary path scalePow10 with
  | .error e => return .error e
  | .ok certs =>
      return verifyHeadContributionCerts certs

/-- Soundly compute conservative per-layer residual amplification constants from a `.nfpt` file. -/
def certifyModelFileGlobal
    (path : System.FilePath)
    (soundnessBits : Nat)
    (inputPath? : Option System.FilePath := none)
    (inputDelta : Rat := 0)
    (partitionDepth : Nat := 0)
    (softmaxMarginLowerBound : Rat := 0)
    (softmaxExpEffort : Nat := defaultSoftmaxExpEffort) : IO (Except String ModelCert) := do
  match ← readModelHeader path with
  | .error e => return .error e
  | .ok (eps, geluTarget) =>
      match ←
          Nfp.Untrusted.SoundCompute.certifyModelFileGlobal
            path eps geluTarget soundnessBits inputPath? inputDelta partitionDepth
            softmaxMarginLowerBound softmaxExpEffort with
      | .error e => return .error e
      | .ok cert =>
          match ← recomputeModelWeightBounds path with
          | .error e =>
              return .error s!"model weight bounds verification failed: {e}"
          | .ok bounds =>
              return verifyModelCert cert eps soundnessBits geluTarget
                bounds.attnValueCoeff bounds.wqOpBoundMax bounds.wkOpBoundMax
                bounds.mlpWinBound bounds.mlpWoutBound
                bounds.ln1MaxAbsGamma bounds.ln1MaxAbsBeta bounds.ln2MaxAbsGamma

/-- Entry point for sound certification (global or local). -/
def certifyModelFile
    (path : System.FilePath)
    (soundnessBits : Nat)
    (inputPath? : Option System.FilePath := none)
    (inputDelta : Rat := 0)
    (partitionDepth : Nat := 0)
    (softmaxMarginLowerBound : Rat := 0)
    (softmaxExpEffort : Nat := defaultSoftmaxExpEffort) : IO (Except String ModelCert) := do
  match ← readModelHeader path with
  | .error e => return .error e
  | .ok (eps, geluTarget) =>
      match ←
          Nfp.Untrusted.SoundCompute.certifyModelFile
            path eps geluTarget soundnessBits inputPath? inputDelta partitionDepth
            softmaxMarginLowerBound softmaxExpEffort with
      | .error e => return .error e
      | .ok cert =>
          match ← recomputeModelWeightBounds path with
          | .error e =>
              return .error s!"model weight bounds verification failed: {e}"
          | .ok bounds =>
              return verifyModelCert cert eps soundnessBits geluTarget
                bounds.attnValueCoeff bounds.wqOpBoundMax bounds.wkOpBoundMax
                bounds.mlpWinBound bounds.mlpWoutBound
                bounds.ln1MaxAbsGamma bounds.ln1MaxAbsBeta bounds.ln2MaxAbsGamma

/-- Compute per-head contribution bounds (global). -/
def certifyHeadBounds
    (path : System.FilePath)
    (scalePow10 : Nat := 9) :
    IO (Except String (Array HeadContributionCert)) := do
  match ← Nfp.Untrusted.SoundCompute.certifyHeadBounds path scalePow10 with
  | .error e => return .error e
  | .ok certs =>
      return verifyHeadContributionCerts certs

/-- Compute local per-head attention contribution bounds. -/
def certifyHeadBoundsLocal
    (path : System.FilePath)
    (inputPath? : Option System.FilePath := none)
    (inputDelta : Rat := 0)
    (soundnessBits : Nat)
    (scalePow10 : Nat := 9) :
    IO (Except String (Array HeadLocalContributionCert)) := do
  match ← readModelEps path with
  | .error e => return .error e
  | .ok eps =>
      match ←
          Nfp.Untrusted.SoundCompute.certifyHeadBoundsLocal
            path eps inputPath? inputDelta soundnessBits scalePow10 with
      | .error e => return .error e
      | .ok certs =>
          return verifyHeadLocalContributionCerts eps soundnessBits certs

/-- Compute local attention pattern bounds for a specific head. -/
def certifyHeadPatternLocal
    (path : System.FilePath)
    (layerIdx headIdx : Nat)
    (inputPath? : Option System.FilePath := none)
    (inputDelta : Rat := 0)
    (soundnessBits : Nat)
    (targetOffset : Int := -1)
    (keyOffset : Int := 0)
    (maxSeqLen : Nat := 256)
    (tightPattern : Bool := false)
    (tightPatternLayers : Nat := 1)
    (perRowPatternLayers : Nat := 0)
    (scalePow10 : Nat := 9)
    (softmaxExpEffort : Nat := defaultSoftmaxExpEffort)
    (causalPattern : Bool := true) :
    IO (Except String HeadPatternCert) := do
  match ← readModelEps path with
  | .error e => return .error e
  | .ok eps =>
      match ←
          Nfp.Untrusted.SoundCompute.certifyHeadPatternLocal
            path layerIdx headIdx eps soundnessBits inputPath? inputDelta targetOffset keyOffset
            maxSeqLen tightPattern tightPatternLayers perRowPatternLayers scalePow10
            softmaxExpEffort causalPattern with
      | .error e => return .error e
      | .ok cert =>
          return verifyHeadPatternCert cert

/-- Compute local best-match pattern bounds for a specific head. -/
def certifyHeadPatternBestMatchLocal
    (path : System.FilePath)
    (layerIdx headIdx : Nat)
    (queryPos? : Option Nat := none)
    (inputPath? : Option System.FilePath := none)
    (inputDelta : Rat := 0)
    (soundnessBits : Nat)
    (targetOffset : Int := -1)
    (keyOffset : Int := 0)
    (maxSeqLen : Nat := 256)
    (tightPattern : Bool := false)
    (tightPatternLayers : Nat := 1)
    (perRowPatternLayers : Nat := 0)
    (useAffine : Bool := false)
    (scalePow10 : Nat := 9)
    (softmaxExpEffort : Nat := defaultSoftmaxExpEffort)
    (causalPattern : Bool := true) :
    IO (Except String HeadBestMatchPatternCert) := do
  match ← readModelEps path with
  | .error e => return .error e
  | .ok eps =>
      match ←
          Nfp.Untrusted.SoundCompute.certifyHeadPatternBestMatchLocal
            path layerIdx headIdx queryPos? eps soundnessBits inputPath? inputDelta targetOffset
              keyOffset maxSeqLen tightPattern tightPatternLayers perRowPatternLayers useAffine
              scalePow10 softmaxExpEffort causalPattern with
      | .error e => return .error e
      | .ok cert =>
          return verifyHeadBestMatchPatternCert cert

/-- Compute local best-match pattern bounds for a sweep of heads. -/
def certifyHeadPatternBestMatchLocalSweep
    (path : System.FilePath)
    (layerIdx headIdx : Nat)
    (inputPath? : Option System.FilePath := none)
    (inputDelta : Rat := 0)
    (soundnessBits : Nat)
    (targetOffset : Int := -1)
    (keyOffset : Int := 0)
    (maxSeqLen : Nat := 256)
    (tightPattern : Bool := false)
    (tightPatternLayers : Nat := 1)
    (perRowPatternLayers : Nat := 0)
    (useAffine : Bool := false)
    (scalePow10 : Nat := 9)
    (softmaxExpEffort : Nat := defaultSoftmaxExpEffort)
    (causalPattern : Bool := true) :
    IO (Except String (Array HeadBestMatchPatternCert)) := do
  match ← readModelEps path with
  | .error e => return .error e
  | .ok eps =>
      match ←
          Nfp.Untrusted.SoundCompute.certifyHeadPatternBestMatchLocalSweep
            path layerIdx headIdx eps soundnessBits inputPath? inputDelta targetOffset keyOffset
            maxSeqLen tightPattern tightPatternLayers perRowPatternLayers useAffine scalePow10
            softmaxExpEffort causalPattern with
      | .error e => return .error e
      | .ok certs =>
          return verifyHeadBestMatchPatternCerts certs

/-- Compute layer-level best-match margin evidence (binary only). -/
def certifyLayerBestMatchMarginLocal
    (path : System.FilePath)
    (layerIdx : Nat)
    (inputPath? : Option System.FilePath := none)
    (inputDelta : Rat := 0)
    (soundnessBits : Nat)
    (targetOffset : Int := -1)
    (keyOffset : Int := 0)
    (maxSeqLen : Nat := 256)
    (tightPattern : Bool := false)
    (tightPatternLayers : Nat := 1)
    (perRowPatternLayers : Nat := 0)
    (scalePow10 : Nat := 9)
    (softmaxExpEffort : Nat := defaultSoftmaxExpEffort)
    (causalPattern : Bool := true) :
    IO (Except String LayerBestMatchMarginCert) := do
  match ← readModelEps path with
  | .error e => return .error e
  | .ok eps =>
      match ←
          Nfp.Untrusted.SoundCompute.certifyLayerBestMatchMarginLocal
            path layerIdx eps soundnessBits inputPath? inputDelta targetOffset keyOffset
            maxSeqLen tightPattern tightPatternLayers perRowPatternLayers scalePow10
            softmaxExpEffort causalPattern with
      | .error e => return .error e
      | .ok cert =>
          return verifyLayerBestMatchMarginCert cert

/-- Soundly compute conservative bounds and tighten them using best-match margin evidence. -/
def certifyModelFileBestMatchMargins
    (path : System.FilePath)
    (soundnessBits : Nat)
    (inputPath? : Option System.FilePath := none)
    (inputDelta : Rat := 0)
    (partitionDepth : Nat := 0)
    (targetOffset : Int := -1)
    (maxSeqLen : Nat := 0)
    (tightPattern : Bool := false)
    (tightPatternLayers : Nat := 1)
    (perRowPatternLayers : Nat := 0)
    (scalePow10 : Nat := 9)
    (softmaxExpEffort : Nat := defaultSoftmaxExpEffort)
    (causalPattern : Bool := true) : IO (Except String ModelCert) := do
  match ← readBinaryModelHeader path with
  | .error e => return .error e
  | .ok hdr =>
      if inputPath?.isNone then
        return .error "best-match margin tightening requires local input"
      let maxSeqLen' := if maxSeqLen = 0 then hdr.seqLen else maxSeqLen
      match ←
          Nfp.Untrusted.SoundCompute.certifyModelFile
            path hdr.eps hdr.geluDerivTarget soundnessBits inputPath? inputDelta partitionDepth
            (softmaxMarginLowerBound := 0) (softmaxExpEffort := softmaxExpEffort) with
      | .error e => return .error e
      | .ok cert =>
          match ← recomputeModelWeightBounds path with
          | .error e =>
              return .error s!"model weight bounds verification failed: {e}"
          | .ok bounds =>
              let mut marginCerts : Array LayerBestMatchMarginCert := Array.mkEmpty hdr.numLayers
              for layerIdx in [:hdr.numLayers] do
                match ←
                    certifyLayerBestMatchMarginLocal path layerIdx
                      (inputPath? := inputPath?) (inputDelta := inputDelta)
                      (soundnessBits := soundnessBits)
                      (targetOffset := targetOffset) (maxSeqLen := maxSeqLen')
                      (tightPattern := tightPattern)
                      (tightPatternLayers := tightPatternLayers)
                      (perRowPatternLayers := perRowPatternLayers)
                      (scalePow10 := scalePow10)
                      (softmaxExpEffort := softmaxExpEffort)
                      (causalPattern := causalPattern) with
                | .error e => return .error e
                | .ok cert => marginCerts := marginCerts.push cert
              return verifyModelCertBestMatchMargins cert hdr.eps soundnessBits hdr.geluDerivTarget
                bounds.attnValueCoeff bounds.wqOpBoundMax bounds.wkOpBoundMax
                bounds.mlpWinBound bounds.mlpWoutBound
                bounds.ln1MaxAbsGamma bounds.ln1MaxAbsBeta bounds.ln2MaxAbsGamma
                marginCerts

/-- Compute local per-head attention contribution bounds tightened by
    best-match pattern evidence. -/
def certifyHeadBoundsLocalBestMatch
    (path : System.FilePath)
    (layerIdx headIdx : Nat)
    (queryPos? : Option Nat := none)
    (inputPath? : Option System.FilePath := none)
    (inputDelta : Rat := 0)
    (soundnessBits : Nat)
    (targetOffset : Int := -1)
    (keyOffset : Int := 0)
    (maxSeqLen : Nat := 256)
    (tightPattern : Bool := false)
    (tightPatternLayers : Nat := 1)
    (perRowPatternLayers : Nat := 0)
    (scalePow10 : Nat := 9)
    (softmaxExpEffort : Nat := defaultSoftmaxExpEffort)
    (causalPattern : Bool := true) :
    IO (Except String HeadLocalContributionCert) := do
  match ← readModelEps path with
  | .error e => return .error e
  | .ok eps =>
      match ←
          certifyHeadBoundsLocal path
            (inputPath? := inputPath?) (inputDelta := inputDelta)
            (soundnessBits := soundnessBits) (scalePow10 := scalePow10) with
      | .error e => return .error e
      | .ok certs =>
          match findHeadLocalContribution certs layerIdx headIdx with
          | .error e => return .error e
          | .ok base =>
              match ←
                  certifyHeadPatternBestMatchLocal path layerIdx headIdx
                    (queryPos? := queryPos?) (inputPath? := inputPath?)
                    (inputDelta := inputDelta) (soundnessBits := soundnessBits)
                    (targetOffset := targetOffset) (keyOffset := keyOffset)
                    (maxSeqLen := maxSeqLen)
                    (tightPattern := tightPattern) (tightPatternLayers := tightPatternLayers)
                    (perRowPatternLayers := perRowPatternLayers)
                    (softmaxExpEffort := softmaxExpEffort)
                    (causalPattern := causalPattern) with
              | .error e => return .error e
              | .ok pattern =>
                  return tightenHeadLocalContributionBestMatch
                    eps soundnessBits base pattern pattern.softmaxExpEffort

/-- Compute local head output lower bounds. -/
def certifyHeadValueLowerBoundLocal
    (path : System.FilePath)
    (layerIdx headIdx : Nat)
    (coord : Nat)
    (inputPath? : Option System.FilePath := none)
    (inputDelta : Rat := 0)
    (soundnessBits : Nat)
    (targetOffset : Int := -1)
    (keyOffset : Int := 0)
    (maxSeqLen : Nat := 256)
    (tightPattern : Bool := false)
    (tightPatternLayers : Nat := 1)
    (perRowPatternLayers : Nat := 0)
    (scalePow10 : Nat := 9)
    (causalPattern : Bool := true) :
    IO (Except String HeadValueLowerBoundCert) := do
  match ← readModelEps path with
  | .error e => return .error e
  | .ok eps =>
      match ←
          Nfp.Untrusted.SoundCompute.certifyHeadValueLowerBoundLocal
            path layerIdx headIdx coord eps soundnessBits inputPath? inputDelta targetOffset
            keyOffset maxSeqLen tightPattern tightPatternLayers perRowPatternLayers scalePow10
            causalPattern with
      | .error e => return .error e
      | .ok cert =>
          return verifyHeadValueLowerBoundCert cert

/-- Compute local head logit-difference lower bounds. -/
def certifyHeadLogitDiffLowerBoundLocal
    (path : System.FilePath)
    (layerIdx headIdx : Nat)
    (targetToken negativeToken : Nat)
    (inputPath? : Option System.FilePath := none)
    (inputDelta : Rat := 0)
    (soundnessBits : Nat)
    (targetOffset : Int := -1)
    (keyOffset : Int := 0)
    (maxSeqLen : Nat := 256)
    (tightPattern : Bool := false)
    (tightPatternLayers : Nat := 1)
    (perRowPatternLayers : Nat := 0)
    (scalePow10 : Nat := 9)
    (causalPattern : Bool := true) :
    IO (Except String HeadLogitDiffLowerBoundCert) := do
  match ← readModelEps path with
  | .error e => return .error e
  | .ok eps =>
      match ←
          Nfp.Untrusted.SoundCompute.certifyHeadLogitDiffLowerBoundLocal
            path layerIdx headIdx targetToken negativeToken eps soundnessBits inputPath? inputDelta
            targetOffset keyOffset maxSeqLen tightPattern tightPatternLayers
            perRowPatternLayers scalePow10 causalPattern with
      | .error e => return .error e
      | .ok cert =>
          return verifyHeadLogitDiffLowerBoundCert cert

/-- Sound induction-head certification (local path). -/
def certifyInductionSound
    (path : System.FilePath)
    (layer1 head1 layer2 head2 : Nat)
    (coord : Nat)
    (inputPath? : Option System.FilePath := none)
    (inputDelta : Rat := 0)
    (soundnessBits : Nat)
    (offset1 : Int := -1)
    (offset2 : Int := -1)
    (keyOffset1 : Int := 0)
    (keyOffset2 : Int := 0)
    (maxSeqLen : Nat := 256)
    (scalePow10 : Nat := 9)
    (tightPattern : Bool := false)
    (tightPatternLayers : Nat := 1)
    (perRowPatternLayers : Nat := 0)
    (targetToken? : Option Nat := none)
    (negativeToken? : Option Nat := none)
    (softmaxExpEffort : Nat := defaultSoftmaxExpEffort)
    (causalPattern : Bool := true) :
    IO (Except String InductionHeadSoundCert) := do
  match ← readModelEps path with
  | .error e => return .error e
  | .ok eps =>
      match ←
          Nfp.Untrusted.SoundCompute.certifyInductionSound
            path layer1 head1 layer2 head2 coord eps soundnessBits inputPath? inputDelta
            offset1 offset2 keyOffset1 keyOffset2 maxSeqLen scalePow10 tightPattern
            tightPatternLayers
            perRowPatternLayers targetToken? negativeToken? softmaxExpEffort causalPattern with
      | .error e => return .error e
      | .ok cert =>
          return verifyInductionHeadSoundCert cert

/-- Sound best-match induction-head certification (local path). -/
def certifyInductionSoundBestMatch
    (path : System.FilePath)
    (layer1 head1 layer2 head2 : Nat)
    (coord : Nat)
    (queryPos? : Option Nat := none)
    (inputPath? : Option System.FilePath := none)
    (inputDelta : Rat := 0)
    (soundnessBits : Nat)
    (offset1 : Int := -1)
    (offset2 : Int := -1)
    (keyOffset1 : Int := 0)
    (keyOffset2 : Int := 0)
    (maxSeqLen : Nat := 256)
    (scalePow10 : Nat := 9)
    (tightPattern : Bool := false)
    (tightPatternLayers : Nat := 1)
    (perRowPatternLayers : Nat := 0)
    (useAffine : Bool := false)
    (iterTighten : Bool := false)
    (targetToken? : Option Nat := none)
    (negativeToken? : Option Nat := none)
    (softmaxExpEffort : Nat := defaultSoftmaxExpEffort)
    (causalPattern : Bool := true) :
    IO (Except String InductionHeadBestMatchSoundCert) := do
  match ← readModelEps path with
  | .error e => return .error e
  | .ok eps =>
      match ←
          Nfp.Untrusted.SoundCompute.certifyInductionSoundBestMatch
            path layer1 head1 layer2 head2 coord queryPos? eps soundnessBits inputPath? inputDelta
            offset1 offset2 keyOffset1 keyOffset2 maxSeqLen scalePow10 tightPattern
            tightPatternLayers perRowPatternLayers useAffine iterTighten targetToken? negativeToken?
            softmaxExpEffort
            causalPattern with
      | .error e => return .error e
      | .ok cert =>
          return verifyInductionHeadBestMatchSoundCert cert

end Nfp.Sound
