-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Mathlib.Data.Finset.Insert
public import Nfp.IO.Parse.Basic
public import Nfp.Model.InductionHead

/-!
Parsing helpers for induction-head input payloads from UTF-8 bytes.
-/

public section

namespace Nfp

namespace IO

namespace Parse

private def kwSeq : ByteArray := "seq".toUTF8
private def kwDModel : ByteArray := "d_model".toUTF8
private def kwDHead : ByteArray := "d_head".toUTF8
private def kwScale : ByteArray := "scale".toUTF8
private def kwActive : ByteArray := "active".toUTF8
private def kwPrev : ByteArray := "prev".toUTF8
private def kwEmbed : ByteArray := "embed".toUTF8
private def kwLnEps : ByteArray := "ln_eps".toUTF8
private def kwLn1Gamma : ByteArray := "ln1_gamma".toUTF8
private def kwLn1Beta : ByteArray := "ln1_beta".toUTF8
private def kwAttnBias : ByteArray := "attn_bias".toUTF8
private def kwMask : ByteArray := "mask".toUTF8
private def kwMaskValue : ByteArray := "mask_value".toUTF8
private def kwCausal : ByteArray := "causal".toUTF8
private def kwNone : ByteArray := "none".toUTF8
private def kwDirection : ByteArray := "direction".toUTF8
private def kwDirectionTarget : ByteArray := "direction-target".toUTF8
private def kwDirectionNegative : ByteArray := "direction-negative".toUTF8

private structure ByteToken where
  start : Nat
  stop : Nat

private def tokenLen (t : ByteToken) : Nat :=
  t.stop - t.start

private def tokenEq (data : ByteArray) (t : ByteToken) (kw : ByteArray) : Bool := Id.run do
  if tokenLen t != kw.size then
    return false
  let mut i := 0
  while i < kw.size do
    if data.get! (t.start + i) != kw.get! i then
      return false
    i := i + 1
  return true

private def parseNatBytesCore (data : ByteArray) (i stop : Nat) (acc : Nat) :
    Except String Nat :=
  if h : i < stop then
    let b := data.get! i
    if b >= 48 && b <= 57 then
      parseNatBytesCore data (i + 1) stop (acc * 10 + (b.toNat - 48))
    else
      Except.error "expected Nat"
  else
    Except.ok acc
termination_by stop - i

private def parseNatBytesSpec (data : ByteArray) (t : ByteToken) : Except String Nat :=
  if tokenLen t = 0 then
    throw "expected Nat"
  else
    parseNatBytesCore data t.start t.stop 0

private def parseNatBytes (data : ByteArray) (t : ByteToken) : Except String Nat :=
  parseNatBytesSpec data t

private theorem parseNatBytes_eq_spec (data : ByteArray) (t : ByteToken) :
    parseNatBytes data t = parseNatBytesSpec data t := by
  rfl

private def parseIntBytesSpec (data : ByteArray) (t : ByteToken) : Except String Int := do
  if tokenLen t = 0 then
    throw "expected Int"
  let first := data.get! t.start
  if first = 45 then
    let t' : ByteToken := { start := t.start + 1, stop := t.stop }
    let n ← parseNatBytesSpec data t'
    return -Int.ofNat n
  else
    let n ← parseNatBytesSpec data t
    return Int.ofNat n

private def parseIntBytes (data : ByteArray) (t : ByteToken) : Except String Int :=
  parseIntBytesSpec data t

private theorem parseIntBytes_eq_spec (data : ByteArray) (t : ByteToken) :
    parseIntBytes data t = parseIntBytesSpec data t := by
  rfl

private def findSlash (data : ByteArray) (i stop : Nat) : Option Nat :=
  if h : i < stop then
    if data.get! i = 47 then
      some i
    else
      findSlash data (i + 1) stop
  else
    none
termination_by stop - i

private def parseRatBytesSpec (data : ByteArray) (t : ByteToken) : Except String Rat := do
  match findSlash data t.start t.stop with
  | none =>
      return ratRoundDown (Rat.ofInt (← parseIntBytesSpec data t))
  | some s =>
      let numTok : ByteToken := { start := t.start, stop := s }
      let denTok : ByteToken := { start := s + 1, stop := t.stop }
      let n ← parseIntBytesSpec data numTok
      let d ← parseNatBytesSpec data denTok
      if d = 0 then
        throw "invalid rational: zero denominator"
      else
        return ratRoundDown (Rat.divInt n (Int.ofNat d))

private def parseRatBytes (data : ByteArray) (t : ByteToken) : Except String Rat :=
  parseRatBytesSpec data t

private theorem parseRatBytes_eq_spec (data : ByteArray) (t : ByteToken) :
    parseRatBytes data t = parseRatBytesSpec data t := by
  rfl

private def nextLineBounds (data : ByteArray) (start : Nat) : Nat × Nat × Nat :=
  Id.run do
    let mut i := start
    let lineStart := start
    while i < data.size do
      let b := data.get! i
      if b == 10 || b == 13 then
        let lineEnd := i
        let mut j := i + 1
        if b == 13 && j < data.size && data.get! j == 10 then
          j := j + 1
        return (j, lineStart, lineEnd)
      i := i + 1
    return (data.size, lineStart, data.size)

private def skipSpaces (data : ByteArray) (i lineEnd : Nat) : Nat :=
  Id.run do
    let mut j := i
    while j < lineEnd do
      let b := data.get! j
      if b == 32 || b == 9 then
        j := j + 1
      else
        break
    return j

private def readToken (data : ByteArray) (i lineEnd : Nat) :
    Option (ByteToken × Nat) :=
  Id.run do
    let j := skipSpaces data i lineEnd
    if j >= lineEnd then
      return none
    let start := j
    let mut k := j
    while k < lineEnd do
      let b := data.get! k
      if b == 32 || b == 9 then
        break
      k := k + 1
    return some ({ start := start, stop := k }, k)

private def expectToken (data : ByteArray) (i lineEnd : Nat) :
    Except String (ByteToken × Nat) := do
  match readToken data i lineEnd with
  | some out => return out
  | none => throw "expected token"

private def ensureNoMoreTokens (data : ByteArray) (i lineEnd : Nat) :
    Except String Unit := do
  let j := skipSpaces data i lineEnd
  if j < lineEnd then
    throw "unrecognized line"

private def parseNatAt (data : ByteArray) (i lineEnd : Nat) :
    Except String (Nat × Nat) := do
  let (tok, i') ← expectToken data i lineEnd
  let n ← parseNatBytes data tok
  return (n, i')

private def parseRatAt (data : ByteArray) (i lineEnd : Nat) :
    Except String (Rat × Nat) := do
  let (tok, i') ← expectToken data i lineEnd
  let r ← parseRatBytes data tok
  return (r, i')

private def setVecEntry (n : Nat) (vec : Array (Option Rat))
    (i : Nat) (v : Rat) :
    Except String (Array (Option Rat)) := do
  if i < n then
    match vec.getD i none with
    | some _ =>
        throw s!"duplicate entry for index={i}"
    | none =>
        let vec' := vec.set! i (some v)
        return vec'
  else
    throw s!"index out of range: i={i}"

private def setMatEntry (rows cols : Nat) (mat : Array (Array (Option Rat)))
    (i j : Nat) (v : Rat) : Except String (Array (Array (Option Rat))) := do
  if i < rows then
    if j < cols then
      let row := mat.getD i #[]
      match row.getD j none with
      | some _ =>
          throw s!"duplicate entry for index=({i}, {j})"
      | none =>
          let row' := row.set! j (some v)
          let mat' := mat.set! i row'
          return mat'
    else
      throw s!"index out of range: j={j}"
  else
    throw s!"index out of range: i={i}"

private def initVecOpt (n : Nat) : Array (Option Rat) :=
  Array.replicate n none

private def initMatOpt (rows cols : Nat) : Array (Array (Option Rat)) :=
  Array.replicate rows (initVecOpt cols)

private def initPrevOpt (n : Nat) : Array (Option (Fin n)) :=
  Array.replicate n none

private def initActiveBits (n : Nat) : Array Bool :=
  Array.replicate n false

private def activeFromBits {n : Nat} (bits : Array Bool) : Finset (Fin n) :=
  (Finset.univ : Finset (Fin n)).filter (fun i => bits.getD i.1 false)

private def arrayAllSome {α : Type} (arr : Array (Option α)) : Bool :=
  (List.range arr.size).all (fun i => (arr.getD i none).isSome)

private def matAllSome {α : Type} (mat : Array (Array (Option α))) : Bool :=
  (List.range mat.size).all (fun i => arrayAllSome (mat.getD i #[]))

private structure HeadParseState (seq dModel dHead : Nat) where
  scale : Option Rat
  activeBits : Array Bool
  activeSeen : Bool
  prev : Array (Option (Fin seq))
  embed : Array (Array (Option Rat))
  lnEps : Option Rat
  ln1Gamma : Array (Option Rat)
  ln1Beta : Array (Option Rat)
  wq : Array (Array (Option Rat))
  bq : Array (Option Rat)
  wk : Array (Array (Option Rat))
  bk : Array (Option Rat)
  wv : Array (Array (Option Rat))
  bv : Array (Option Rat)
  wo : Array (Array (Option Rat))
  attnBias : Array (Option Rat)
  maskCausal : Option Bool
  maskValue : Option Rat
  directionTarget : Option Nat
  directionNegative : Option Nat
  direction : Array (Option Rat)

private def initHeadState (seq dModel dHead : Nat) : HeadParseState seq dModel dHead :=
  { scale := none
    activeBits := initActiveBits seq
    activeSeen := false
    prev := initPrevOpt seq
    embed := initMatOpt seq dModel
    lnEps := none
    ln1Gamma := initVecOpt dModel
    ln1Beta := initVecOpt dModel
    wq := initMatOpt dModel dHead
    bq := initVecOpt dHead
    wk := initMatOpt dModel dHead
    bk := initVecOpt dHead
    wv := initMatOpt dModel dHead
    bv := initVecOpt dHead
    wo := initMatOpt dModel dHead
    attnBias := initVecOpt dModel
    maskCausal := none
    maskValue := none
    directionTarget := none
    directionNegative := none
    direction := initVecOpt dModel }

private def setHeadActive {seq dModel dHead : Nat}
    (st : HeadParseState seq dModel dHead) (q : Nat) :
    Except String (HeadParseState seq dModel dHead) := do
  if q < seq then
    return { st with activeBits := st.activeBits.set! q true, activeSeen := true }
  else
    throw s!"active index out of range: q={q}"

private def setHeadPrev {seq dModel dHead : Nat}
    (st : HeadParseState seq dModel dHead) (q k : Nat) :
    Except String (HeadParseState seq dModel dHead) := do
  if q < seq then
    if hk : k < seq then
      let kFin : Fin seq := ⟨k, hk⟩
      match st.prev.getD q none with
      | some _ =>
          throw s!"duplicate prev entry for q={q}"
      | none =>
          return { st with prev := st.prev.set! q (some kFin) }
    else
      throw s!"prev index out of range: k={k}"
  else
    throw s!"prev index out of range: q={q}"

private def parseHeadLine {seq dModel dHead : Nat} (st : HeadParseState seq dModel dHead)
    (tokens : List String) : Except String (HeadParseState seq dModel dHead) := do
  match tokens with
  | ["scale", val] =>
      if st.scale.isSome then
        throw "duplicate scale entry"
      else
        return { st with scale := some (← parseRat val) }
  | ["active", q] =>
      setHeadActive st (← parseNat q)
  | ["prev", q, k] =>
      setHeadPrev st (← parseNat q) (← parseNat k)
  | ["embed", q, d, val] =>
      let mat ←
        setMatEntry seq dModel st.embed (← parseNat q) (← parseNat d) (← parseRat val)
      return { st with embed := mat }
  | ["ln_eps", val] =>
      if st.lnEps.isSome then
        throw "duplicate ln_eps entry"
      else
        return { st with lnEps := some (← parseRat val) }
  | ["ln1_gamma", d, val] =>
      let vec ← setVecEntry dModel st.ln1Gamma (← parseNat d) (← parseRat val)
      return { st with ln1Gamma := vec }
  | ["ln1_beta", d, val] =>
      let vec ← setVecEntry dModel st.ln1Beta (← parseNat d) (← parseRat val)
      return { st with ln1Beta := vec }
  | ["wq", i, j, val] =>
      let mat ←
        setMatEntry dModel dHead st.wq (← parseNat i) (← parseNat j) (← parseRat val)
      return { st with wq := mat }
  | ["bq", j, val] =>
      let vec ← setVecEntry dHead st.bq (← parseNat j) (← parseRat val)
      return { st with bq := vec }
  | ["wk", i, j, val] =>
      let mat ←
        setMatEntry dModel dHead st.wk (← parseNat i) (← parseNat j) (← parseRat val)
      return { st with wk := mat }
  | ["bk", j, val] =>
      let vec ← setVecEntry dHead st.bk (← parseNat j) (← parseRat val)
      return { st with bk := vec }
  | ["wv", i, j, val] =>
      let mat ←
        setMatEntry dModel dHead st.wv (← parseNat i) (← parseNat j) (← parseRat val)
      return { st with wv := mat }
  | ["bv", j, val] =>
      let vec ← setVecEntry dHead st.bv (← parseNat j) (← parseRat val)
      return { st with bv := vec }
  | ["wo", i, j, val] =>
      let mat ←
        setMatEntry dModel dHead st.wo (← parseNat i) (← parseNat j) (← parseRat val)
      return { st with wo := mat }
  | ["attn_bias", d, val] =>
      let vec ← setVecEntry dModel st.attnBias (← parseNat d) (← parseRat val)
      return { st with attnBias := vec }
  | ["mask", kind] =>
      if st.maskCausal.isSome then
        throw "duplicate mask entry"
      else
        match kind with
        | "causal" => return { st with maskCausal := some true }
        | "none" => return { st with maskCausal := some false }
        | _ => throw "mask must be 'causal' or 'none'"
  | ["mask_value", val] =>
      if st.maskValue.isSome then
        throw "duplicate mask_value entry"
      else
        return { st with maskValue := some (← parseRat val) }
  | ["direction-target", tok] =>
      if st.directionTarget.isSome then
        throw "duplicate direction-target entry"
      else
        return { st with directionTarget := some (← parseNat tok) }
  | ["direction-negative", tok] =>
      if st.directionNegative.isSome then
        throw "duplicate direction-negative entry"
      else
        return { st with directionNegative := some (← parseNat tok) }
  | ["direction", d, val] =>
      let vec ← setVecEntry dModel st.direction (← parseNat d) (← parseRat val)
      return { st with direction := vec }
  | _ =>
      throw s!"unrecognized line: '{String.intercalate " " tokens}'"

private def parseHeadLineBytes {seq dModel dHead : Nat} (data : ByteArray)
    (st : HeadParseState seq dModel dHead) (lineStart lineEnd : Nat) :
    Except String (HeadParseState seq dModel dHead) := do
  let i0 := skipSpaces data lineStart lineEnd
  if i0 >= lineEnd then
    return st
  if data.get! i0 = 35 then
    return st
  match readToken data i0 lineEnd with
  | none => return st
  | some (t0, i1) =>
      let len := tokenLen t0
      let b0 := data.get! t0.start
      match b0 with
      | 115 => -- s
          if len = kwSeq.size && tokenEq data t0 kwSeq then
            return st
          else if len = kwScale.size && tokenEq data t0 kwScale then
            if st.scale.isSome then
              throw "duplicate scale entry"
            else
              let (t1, i2) ← expectToken data i1 lineEnd
              ensureNoMoreTokens data i2 lineEnd
              return { st with scale := some (← parseRatBytes data t1) }
          else
            throw "unrecognized line"
      | 97 => -- a
          if len = kwActive.size && tokenEq data t0 kwActive then
            let (q, i2) ← parseNatAt data i1 lineEnd
            ensureNoMoreTokens data i2 lineEnd
            setHeadActive st q
          else if len = kwAttnBias.size && tokenEq data t0 kwAttnBias then
            let (d, i2) ← parseNatAt data i1 lineEnd
            let (v, i3) ← parseRatAt data i2 lineEnd
            ensureNoMoreTokens data i3 lineEnd
            let vec ← setVecEntry dModel st.attnBias d v
            return { st with attnBias := vec }
          else
            throw "unrecognized line"
      | 112 => -- p
          if len = kwPrev.size && tokenEq data t0 kwPrev then
            let (q, i2) ← parseNatAt data i1 lineEnd
            let (k, i3) ← parseNatAt data i2 lineEnd
            ensureNoMoreTokens data i3 lineEnd
            setHeadPrev st q k
          else
            throw "unrecognized line"
      | 101 => -- e
          if len = kwEmbed.size && tokenEq data t0 kwEmbed then
            let (q, i2) ← parseNatAt data i1 lineEnd
            let (d, i3) ← parseNatAt data i2 lineEnd
            let (v, i4) ← parseRatAt data i3 lineEnd
            ensureNoMoreTokens data i4 lineEnd
            let mat ← setMatEntry seq dModel st.embed q d v
            return { st with embed := mat }
          else
            throw "unrecognized line"
      | 108 => -- l
          if len = kwLnEps.size && tokenEq data t0 kwLnEps then
            if st.lnEps.isSome then
              throw "duplicate ln_eps entry"
            else
              let (v, i2) ← parseRatAt data i1 lineEnd
              ensureNoMoreTokens data i2 lineEnd
              return { st with lnEps := some v }
          else if len = kwLn1Gamma.size && tokenEq data t0 kwLn1Gamma then
            let (d, i2) ← parseNatAt data i1 lineEnd
            let (v, i3) ← parseRatAt data i2 lineEnd
            ensureNoMoreTokens data i3 lineEnd
            let vec ← setVecEntry dModel st.ln1Gamma d v
            return { st with ln1Gamma := vec }
          else if len = kwLn1Beta.size && tokenEq data t0 kwLn1Beta then
            let (d, i2) ← parseNatAt data i1 lineEnd
            let (v, i3) ← parseRatAt data i2 lineEnd
            ensureNoMoreTokens data i3 lineEnd
            let vec ← setVecEntry dModel st.ln1Beta d v
            return { st with ln1Beta := vec }
          else
            throw "unrecognized line"
      | 119 => -- w
          if len = 2 then
            let b1 := data.get! (t0.start + 1)
            let (i, i2) ← parseNatAt data i1 lineEnd
            let (j, i3) ← parseNatAt data i2 lineEnd
            let (v, i4) ← parseRatAt data i3 lineEnd
            ensureNoMoreTokens data i4 lineEnd
            if b1 = 113 then
              let mat ← setMatEntry dModel dHead st.wq i j v
              return { st with wq := mat }
            else if b1 = 107 then
              let mat ← setMatEntry dModel dHead st.wk i j v
              return { st with wk := mat }
            else if b1 = 118 then
              let mat ← setMatEntry dModel dHead st.wv i j v
              return { st with wv := mat }
            else if b1 = 111 then
              let mat ← setMatEntry dModel dHead st.wo i j v
              return { st with wo := mat }
            else
              throw "unrecognized line"
          else
            throw "unrecognized line"
      | 98 => -- b
          if len = 2 then
            let b1 := data.get! (t0.start + 1)
            let (j, i2) ← parseNatAt data i1 lineEnd
            let (v, i3) ← parseRatAt data i2 lineEnd
            ensureNoMoreTokens data i3 lineEnd
            if b1 = 113 then
              let vec ← setVecEntry dHead st.bq j v
              return { st with bq := vec }
            else if b1 = 107 then
              let vec ← setVecEntry dHead st.bk j v
              return { st with bk := vec }
            else if b1 = 118 then
              let vec ← setVecEntry dHead st.bv j v
              return { st with bv := vec }
            else
              throw "unrecognized line"
          else
            throw "unrecognized line"
      | 109 => -- m
          if len = kwMask.size && tokenEq data t0 kwMask then
            if st.maskCausal.isSome then
              throw "duplicate mask entry"
            else
              let (t1, i2) ← expectToken data i1 lineEnd
              ensureNoMoreTokens data i2 lineEnd
              if tokenEq data t1 kwCausal then
                return { st with maskCausal := some true }
              else if tokenEq data t1 kwNone then
                return { st with maskCausal := some false }
              else
                throw "mask must be 'causal' or 'none'"
          else if len = kwMaskValue.size && tokenEq data t0 kwMaskValue then
            if st.maskValue.isSome then
              throw "duplicate mask_value entry"
            else
              let (v, i2) ← parseRatAt data i1 lineEnd
              ensureNoMoreTokens data i2 lineEnd
              return { st with maskValue := some v }
          else
            throw "unrecognized line"
      | 100 => -- d
          if len = kwDModel.size && tokenEq data t0 kwDModel then
            return st
          else if len = kwDHead.size && tokenEq data t0 kwDHead then
            return st
          else if len = kwDirection.size && tokenEq data t0 kwDirection then
            let (d, i2) ← parseNatAt data i1 lineEnd
            let (v, i3) ← parseRatAt data i2 lineEnd
            ensureNoMoreTokens data i3 lineEnd
            let vec ← setVecEntry dModel st.direction d v
            return { st with direction := vec }
          else if len = kwDirectionTarget.size && tokenEq data t0 kwDirectionTarget then
            if st.directionTarget.isSome then
              throw "duplicate direction-target entry"
            else
              let (v, i2) ← parseNatAt data i1 lineEnd
              ensureNoMoreTokens data i2 lineEnd
              return { st with directionTarget := some v }
          else if len = kwDirectionNegative.size && tokenEq data t0 kwDirectionNegative then
            if st.directionNegative.isSome then
              throw "duplicate direction-negative entry"
            else
              let (v, i2) ← parseNatAt data i1 lineEnd
              ensureNoMoreTokens data i2 lineEnd
              return { st with directionNegative := some v }
          else
            throw "unrecognized line"
      | _ =>
          throw "unrecognized line"

private def parseHeaderLineBytes (data : ByteArray) (lineStart lineEnd : Nat)
    (seq? dModel? dHead? : Option Nat) :
    Except String (Option Nat × Option Nat × Option Nat) := do
  let i0 := skipSpaces data lineStart lineEnd
  if i0 >= lineEnd then
    return (seq?, dModel?, dHead?)
  if data.get! i0 = 35 then
    return (seq?, dModel?, dHead?)
  match readToken data i0 lineEnd with
  | none => return (seq?, dModel?, dHead?)
  | some (t0, i1) =>
      if tokenEq data t0 kwSeq then
        let (v, i2) ← parseNatAt data i1 lineEnd
        ensureNoMoreTokens data i2 lineEnd
        if seq?.isSome then
          throw "duplicate seq entry"
        else
          return (some v, dModel?, dHead?)
      else if tokenEq data t0 kwDModel then
        let (v, i2) ← parseNatAt data i1 lineEnd
        ensureNoMoreTokens data i2 lineEnd
        if dModel?.isSome then
          throw "duplicate d_model entry"
        else
          return (seq?, some v, dHead?)
      else if tokenEq data t0 kwDHead then
        let (v, i2) ← parseNatAt data i1 lineEnd
        ensureNoMoreTokens data i2 lineEnd
        if dHead?.isSome then
          throw "duplicate d_head entry"
        else
          return (seq?, dModel?, some v)
      else
        return (seq?, dModel?, dHead?)

private def finalizeHeadState {seq dModel dHead : Nat} (hpos : 0 < seq)
    (st : HeadParseState seq dModel dHead) :
    Except String (Model.InductionHeadInputs seq dModel dHead) := do
  let scale ←
    match st.scale with
    | some v => pure v
    | none => throw "missing scale entry"
  if !arrayAllSome st.prev then
    throw "missing prev entries"
  if !matAllSome st.embed then
    throw "missing embed entries"
  let lnEps ←
    match st.lnEps with
    | some v => pure v
    | none => throw "missing ln_eps entry"
  if !arrayAllSome st.ln1Gamma then
    throw "missing ln1_gamma entries"
  if !arrayAllSome st.ln1Beta then
    throw "missing ln1_beta entries"
  if !matAllSome st.wq then
    throw "missing wq entries"
  if !arrayAllSome st.bq then
    throw "missing bq entries"
  if !matAllSome st.wk then
    throw "missing wk entries"
  if !arrayAllSome st.bk then
    throw "missing bk entries"
  if !matAllSome st.wv then
    throw "missing wv entries"
  if !arrayAllSome st.bv then
    throw "missing bv entries"
  if !matAllSome st.wo then
    throw "missing wo entries"
  if !arrayAllSome st.attnBias then
    throw "missing attn_bias entries"
  if !arrayAllSome st.direction then
    throw "missing direction entries"
  let directionSpec ←
    match st.directionTarget, st.directionNegative with
    | some target, some negative => pure { target := target, negative := negative }
    | _, _ =>
        throw "direction metadata requires both direction-target and direction-negative"
  let defaultPrev : Fin seq := ⟨0, hpos⟩
  let prevFun : Fin seq → Fin seq := fun q =>
    (st.prev.getD q.1 none).getD defaultPrev
  let embedArr : Array (Array Rat) :=
    st.embed.map (fun row => row.map (fun v => v.getD 0))
  let ln1GammaArr : Array Rat :=
    st.ln1Gamma.map (fun v => v.getD 0)
  let ln1BetaArr : Array Rat :=
    st.ln1Beta.map (fun v => v.getD 0)
  let wqArr : Array (Array Rat) :=
    st.wq.map (fun row => row.map (fun v => v.getD 0))
  let bqArr : Array Rat :=
    st.bq.map (fun v => v.getD 0)
  let wkArr : Array (Array Rat) :=
    st.wk.map (fun row => row.map (fun v => v.getD 0))
  let bkArr : Array Rat :=
    st.bk.map (fun v => v.getD 0)
  let wvArr : Array (Array Rat) :=
    st.wv.map (fun row => row.map (fun v => v.getD 0))
  let bvArr : Array Rat :=
    st.bv.map (fun v => v.getD 0)
  let woArr : Array (Array Rat) :=
    st.wo.map (fun row => row.map (fun v => v.getD 0))
  let attnBiasArr : Array Rat :=
    st.attnBias.map (fun v => v.getD 0)
  let directionArr : Array Rat :=
    st.direction.map (fun v => v.getD 0)
  let embedFun : Fin seq → Fin dModel → Rat := fun q d =>
    (embedArr.getD q.1 #[]).getD d.1 0
  let ln1GammaFun : Fin dModel → Rat := fun d =>
    ln1GammaArr.getD d.1 0
  let ln1BetaFun : Fin dModel → Rat := fun d =>
    ln1BetaArr.getD d.1 0
  let wqFun : Fin dModel → Fin dHead → Rat := fun i j =>
    (wqArr.getD i.1 #[]).getD j.1 0
  let bqFun : Fin dHead → Rat := fun j =>
    bqArr.getD j.1 0
  let wkFun : Fin dModel → Fin dHead → Rat := fun i j =>
    (wkArr.getD i.1 #[]).getD j.1 0
  let bkFun : Fin dHead → Rat := fun j =>
    bkArr.getD j.1 0
  let wvFun : Fin dModel → Fin dHead → Rat := fun i j =>
    (wvArr.getD i.1 #[]).getD j.1 0
  let bvFun : Fin dHead → Rat := fun j =>
    bvArr.getD j.1 0
  let woFun : Fin dModel → Fin dHead → Rat := fun i j =>
    (woArr.getD i.1 #[]).getD j.1 0
  let attnBiasFun : Fin dModel → Rat := fun d =>
    attnBiasArr.getD d.1 0
  let maskCausal := st.maskCausal.getD false
  let maskValue :=
    match st.maskValue with
    | some v => v
    | none => if maskCausal then (-10000 : Rat) else 0
  let directionFun : Fin dModel → Rat := fun d =>
    directionArr.getD d.1 0
  let active :=
    if st.activeSeen then
      activeFromBits st.activeBits
    else
      (Finset.univ : Finset (Fin seq)).erase defaultPrev
  pure
    { scale := scale
      active := active
      prev := prevFun
      embed := embedFun
      lnEps := lnEps
      ln1Gamma := ln1GammaFun
      ln1Beta := ln1BetaFun
      wq := wqFun
      bq := bqFun
      wk := wkFun
      bk := bkFun
      wv := wvFun
      bv := bvFun
      wo := woFun
      attnBias := attnBiasFun
      maskCausal := maskCausal
      maskValue := maskValue
      directionSpec := directionSpec
      direction := directionFun }

/-- Parse a raw induction head input payload from UTF-8 bytes. -/
def parseInductionHeadInputsBytes (data : ByteArray) :
    Except String (Sigma (fun seq =>
      Sigma (fun dModel => Sigma (fun dHead => Model.InductionHeadInputs seq dModel dHead)))) := do
  let mut seq? : Option Nat := none
  let mut dModel? : Option Nat := none
  let mut dHead? : Option Nat := none
  let mut i := 0
  let mut afterDims := 0
  let mut haveDims := false
  while i < data.size && !haveDims do
    let (i', lineStart, lineEnd) := nextLineBounds data i
    i := i'
    afterDims := i'
    let (seqNew, dModelNew, dHeadNew) ←
      parseHeaderLineBytes data lineStart lineEnd seq? dModel? dHead?
    seq? := seqNew
    dModel? := dModelNew
    dHead? := dHeadNew
    if seq?.isSome && dModel?.isSome && dHead?.isSome then
      haveDims := true
  let seq ←
    match seq? with
    | some v => pure v
    | none => throw "missing seq entry"
  let dModel ←
    match dModel? with
    | some v => pure v
    | none => throw "missing d_model entry"
  let dHead ←
    match dHead? with
    | some v => pure v
    | none => throw "missing d_head entry"
  match seq, dModel, dHead with
  | 0, _, _ => throw "seq must be positive"
  | _, 0, _ => throw "d_model must be positive"
  | _, _, 0 => throw "d_head must be positive"
  | Nat.succ n, Nat.succ m, Nat.succ h =>
      let seq := Nat.succ n
      let dModel := Nat.succ m
      let dHead := Nat.succ h
      let hpos : 0 < seq := Nat.succ_pos n
      let st0 : HeadParseState seq dModel dHead := initHeadState seq dModel dHead
      let mut st := st0
      let mut j := afterDims
      while j < data.size do
        let (j', lineStart, lineEnd) := nextLineBounds data j
        j := j'
        st ← parseHeadLineBytes data st lineStart lineEnd
      let inputs ← finalizeHeadState hpos st
      return ⟨seq, ⟨dModel, ⟨dHead, inputs⟩⟩⟩

end Parse

end IO

end Nfp
