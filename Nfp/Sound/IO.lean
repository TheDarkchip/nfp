-- SPDX-License-Identifier: AGPL-3.0-or-later

import Std
import Nfp.Sound.Cert
import Nfp.Sound.HeadCert
import Nfp.Sound.ModelHeader
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

/-- Compute weight-only per-head contribution bounds from a binary `.nfpt`. -/
def certifyHeadBoundsBinary
    (path : System.FilePath)
    (scalePow10 : Nat := 9) :
    IO (Except String (Array HeadContributionCert)) := do
  match ← Nfp.Untrusted.SoundCompute.certifyHeadBoundsBinary path scalePow10 with
  | .error e => return .error e
  | .ok certs =>
      let ok := certs.foldl (fun acc c => acc && c.check) true
      if ok then
        return .ok certs
      return .error "head contribution certificate failed internal checks"

/-- Soundly compute conservative per-layer residual amplification constants from a `.nfpt` file. -/
def certifyModelFileGlobal
    (path : System.FilePath)
    (soundnessBits : Nat)
    (inputPath? : Option System.FilePath := none)
    (inputDelta : Rat := 0) : IO (Except String ModelCert) := do
  match ← readModelHeader path with
  | .error e => return .error e
  | .ok (eps, geluTarget) =>
      match ←
          Nfp.Untrusted.SoundCompute.certifyModelFileGlobal
            path eps geluTarget soundnessBits inputPath? inputDelta with
      | .error e => return .error e
      | .ok cert =>
          if cert.eps ≠ eps then
            return .error "model header eps mismatch"
          if cert.soundnessBits ≠ soundnessBits then
            return .error "soundness bits mismatch"
          if cert.geluDerivTarget ≠ geluTarget then
            return .error "model header gelu_kind mismatch"
          if cert.check then
            return .ok cert
          return .error "sound certificate failed internal consistency checks"

/-- Entry point for sound certification (global or local). -/
def certifyModelFile
    (path : System.FilePath)
    (soundnessBits : Nat)
    (inputPath? : Option System.FilePath := none)
    (inputDelta : Rat := 0) : IO (Except String ModelCert) := do
  match ← readModelHeader path with
  | .error e => return .error e
  | .ok (eps, geluTarget) =>
      match ←
          Nfp.Untrusted.SoundCompute.certifyModelFile
            path eps geluTarget soundnessBits inputPath? inputDelta with
      | .error e => return .error e
      | .ok cert =>
          if cert.eps ≠ eps then
            return .error "model header eps mismatch"
          if cert.soundnessBits ≠ soundnessBits then
            return .error "soundness bits mismatch"
          if cert.geluDerivTarget ≠ geluTarget then
            return .error "model header gelu_kind mismatch"
          if cert.check then
            return .ok cert
          return .error "sound certificate failed internal consistency checks"

/-- Compute per-head contribution bounds (global). -/
def certifyHeadBounds
    (path : System.FilePath)
    (scalePow10 : Nat := 9) :
    IO (Except String (Array HeadContributionCert)) := do
  match ← Nfp.Untrusted.SoundCompute.certifyHeadBounds path scalePow10 with
  | .error e => return .error e
  | .ok certs =>
      let ok := certs.foldl (fun acc c => acc && c.check) true
      if ok then
        return .ok certs
      return .error "head contribution certificate failed internal checks"

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
          let ok :=
            certs.foldl (fun acc c =>
              acc && c.soundnessBits = soundnessBits && c.check eps) true
          if ok then
            return .ok certs
          return .error "local head contribution certificate failed internal checks"

/-- Compute local attention pattern bounds for a specific head. -/
def certifyHeadPatternLocal
    (path : System.FilePath)
    (layerIdx headIdx : Nat)
    (inputPath? : Option System.FilePath := none)
    (inputDelta : Rat := 0)
    (soundnessBits : Nat)
    (targetOffset : Int := -1)
    (maxSeqLen : Nat := 256)
    (tightPattern : Bool := false)
    (tightPatternLayers : Nat := 1)
    (perRowPatternLayers : Nat := 0)
    (scalePow10 : Nat := 9) :
    IO (Except String HeadPatternCert) := do
  match ← readModelEps path with
  | .error e => return .error e
  | .ok eps =>
      match ←
          Nfp.Untrusted.SoundCompute.certifyHeadPatternLocal
            path layerIdx headIdx eps soundnessBits inputPath? inputDelta targetOffset maxSeqLen
            tightPattern tightPatternLayers perRowPatternLayers scalePow10 with
      | .error e => return .error e
      | .ok cert =>
          if cert.check then
            return .ok cert
          return .error "head pattern certificate failed internal checks"

/-- Compute local best-match pattern bounds for a specific head. -/
def certifyHeadPatternBestMatchLocal
    (path : System.FilePath)
    (layerIdx headIdx : Nat)
    (queryPos? : Option Nat := none)
    (inputPath? : Option System.FilePath := none)
    (inputDelta : Rat := 0)
    (soundnessBits : Nat)
    (targetOffset : Int := -1)
    (maxSeqLen : Nat := 256)
    (tightPattern : Bool := false)
    (tightPatternLayers : Nat := 1)
    (perRowPatternLayers : Nat := 0)
    (scalePow10 : Nat := 9) :
    IO (Except String HeadBestMatchPatternCert) := do
  match ← readModelEps path with
  | .error e => return .error e
  | .ok eps =>
      match ←
          Nfp.Untrusted.SoundCompute.certifyHeadPatternBestMatchLocal
            path layerIdx headIdx queryPos? eps soundnessBits inputPath? inputDelta targetOffset
              maxSeqLen
            tightPattern tightPatternLayers perRowPatternLayers scalePow10 with
      | .error e => return .error e
      | .ok cert =>
          if cert.check then
            return .ok cert
          return .error "head best-match pattern certificate failed internal checks"

/-- Compute local best-match pattern bounds for a sweep of heads. -/
def certifyHeadPatternBestMatchLocalSweep
    (path : System.FilePath)
    (layerIdx headIdx : Nat)
    (inputPath? : Option System.FilePath := none)
    (inputDelta : Rat := 0)
    (soundnessBits : Nat)
    (targetOffset : Int := -1)
    (maxSeqLen : Nat := 256)
    (tightPattern : Bool := false)
    (tightPatternLayers : Nat := 1)
    (perRowPatternLayers : Nat := 0)
    (scalePow10 : Nat := 9) :
    IO (Except String (Array HeadBestMatchPatternCert)) := do
  match ← readModelEps path with
  | .error e => return .error e
  | .ok eps =>
      match ←
          Nfp.Untrusted.SoundCompute.certifyHeadPatternBestMatchLocalSweep
            path layerIdx headIdx eps soundnessBits inputPath? inputDelta targetOffset maxSeqLen
            tightPattern tightPatternLayers perRowPatternLayers scalePow10 with
      | .error e => return .error e
      | .ok certs =>
          let ok := certs.foldl (fun acc c => acc && c.check) true
          if ok then
            return .ok certs
          return .error "head best-match sweep certificate failed internal checks"

/-- Compute local head output lower bounds. -/
def certifyHeadValueLowerBoundLocal
    (path : System.FilePath)
    (layerIdx headIdx : Nat)
    (coord : Nat)
    (inputPath? : Option System.FilePath := none)
    (inputDelta : Rat := 0)
    (soundnessBits : Nat)
    (targetOffset : Int := -1)
    (maxSeqLen : Nat := 256)
    (tightPattern : Bool := false)
    (tightPatternLayers : Nat := 1)
    (perRowPatternLayers : Nat := 0)
    (scalePow10 : Nat := 9) :
    IO (Except String HeadValueLowerBoundCert) := do
  match ← readModelEps path with
  | .error e => return .error e
  | .ok eps =>
      match ←
          Nfp.Untrusted.SoundCompute.certifyHeadValueLowerBoundLocal
            path layerIdx headIdx coord eps soundnessBits inputPath? inputDelta targetOffset
            maxSeqLen tightPattern tightPatternLayers perRowPatternLayers scalePow10 with
      | .error e => return .error e
      | .ok cert =>
          if cert.check then
            return .ok cert
          return .error "head value lower bound certificate failed internal checks"

/-- Compute local head logit-difference lower bounds. -/
def certifyHeadLogitDiffLowerBoundLocal
    (path : System.FilePath)
    (layerIdx headIdx : Nat)
    (targetToken negativeToken : Nat)
    (inputPath? : Option System.FilePath := none)
    (inputDelta : Rat := 0)
    (soundnessBits : Nat)
    (targetOffset : Int := -1)
    (maxSeqLen : Nat := 256)
    (tightPattern : Bool := false)
    (tightPatternLayers : Nat := 1)
    (perRowPatternLayers : Nat := 0)
    (scalePow10 : Nat := 9) :
    IO (Except String HeadLogitDiffLowerBoundCert) := do
  match ← readModelEps path with
  | .error e => return .error e
  | .ok eps =>
      match ←
          Nfp.Untrusted.SoundCompute.certifyHeadLogitDiffLowerBoundLocal
            path layerIdx headIdx targetToken negativeToken eps soundnessBits inputPath? inputDelta
            targetOffset maxSeqLen tightPattern tightPatternLayers
            perRowPatternLayers scalePow10 with
      | .error e => return .error e
      | .ok cert =>
          if cert.check then
            return .ok cert
      return .error "head logit-diff lower bound certificate failed internal checks"

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
    (maxSeqLen : Nat := 256)
    (scalePow10 : Nat := 9)
    (tightPattern : Bool := false)
    (tightPatternLayers : Nat := 1)
    (perRowPatternLayers : Nat := 0)
    (targetToken? : Option Nat := none)
    (negativeToken? : Option Nat := none) :
    IO (Except String InductionHeadSoundCert) := do
  match ← readModelEps path with
  | .error e => return .error e
  | .ok eps =>
      match ←
          Nfp.Untrusted.SoundCompute.certifyInductionSound
            path layer1 head1 layer2 head2 coord eps soundnessBits inputPath? inputDelta
            offset1 offset2 maxSeqLen scalePow10 tightPattern tightPatternLayers
            perRowPatternLayers targetToken? negativeToken? with
      | .error e => return .error e
      | .ok cert =>
          if cert.check then
            return .ok cert
          return .error "induction head certificate failed internal checks"

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
    (maxSeqLen : Nat := 256)
    (scalePow10 : Nat := 9)
    (tightPattern : Bool := false)
    (tightPatternLayers : Nat := 1)
    (perRowPatternLayers : Nat := 0)
    (targetToken? : Option Nat := none)
    (negativeToken? : Option Nat := none) :
    IO (Except String InductionHeadBestMatchSoundCert) := do
  match ← readModelEps path with
  | .error e => return .error e
  | .ok eps =>
      match ←
          Nfp.Untrusted.SoundCompute.certifyInductionSoundBestMatch
            path layer1 head1 layer2 head2 coord queryPos? eps soundnessBits inputPath? inputDelta
            offset1 offset2 maxSeqLen scalePow10 tightPattern
            tightPatternLayers perRowPatternLayers targetToken? negativeToken? with
      | .error e => return .error e
      | .ok cert =>
          if cert.check then
            return .ok cert
          return .error "best-match induction head certificate failed internal checks"

end Nfp.Sound
