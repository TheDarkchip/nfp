-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Data.List.Range
import Nfp.IO.Timing
import Nfp.Sound.Bounds.MatrixNorm
import Nfp.Sound.Linear.FinFold

/-!
Microbenchmarks for dyadic arithmetic and caching strategies.
-/

namespace Nfp

namespace IO

open Sound

private def benchItersFor (base : Nat) (n : Nat) : Nat :=
  let scale := max 1 (n / 64)
  max 1 (base / scale)

private def mkDyadic (num den : Nat) (neg : Bool) : Dyadic :=
  let n : Int := Int.ofNat (num + 1)
  let d : Int := Int.ofNat (den + 1)
  let q : Rat := Rat.divInt (if neg then -n else n) d
  dyadicOfRatDown q

private def mkVecDyadic (n : Nat) (seed : Nat) (salt : Nat) (negEvery : Nat) : Fin n → Dyadic := fun i =>
  let idx := i.1 + seed + salt
  let neg := (idx % negEvery) = 0
  mkDyadic (idx % 97) (idx % 89) neg

private def mkInterval (n : Nat) (seed : Nat) :
    (Fin n → Dyadic) × (Fin n → Dyadic) × (Fin n → Dyadic) :=
  let v : Fin n → Dyadic := mkVecDyadic n seed 0 2
  let base : Fin n → Dyadic := mkVecDyadic n seed 13 3
  let lo : Fin n → Dyadic := fun i => base i - 1
  let hi : Fin n → Dyadic := fun i => base i + 1
  (v, lo, hi)

private def benchLoop (label : String) (iters : Nat) (act : Unit → Dyadic) : IO Unit := do
  let t0 ← monoUsNow
  let mut last : Dyadic := 0
  for _ in List.range iters do
    last := act ()
  let t1 ← monoUsNow
  let total := t1 - t0
  let avg := total / max 1 iters
  IO.println s!"bench: {label} iters={iters} total={total} us avg={avg} us last={last}"

private def benchDotInterval (n iters seed : Nat) : IO Unit := do
  let (v, lo, hi) := mkInterval n seed
  let labelBase := s!"n={n}"
  benchLoop s!"dotIntervalLower {labelBase}" iters (fun () =>
    Sound.Bounds.dotIntervalLower v lo hi)
  benchLoop s!"dotIntervalLowerCommonDen {labelBase}" iters (fun () =>
    Sound.Bounds.dotIntervalLowerCommonDen v lo hi)
  benchLoop s!"dotIntervalLowerCachedRat {labelBase}" iters (fun () =>
    Sound.Bounds.dotIntervalLowerCachedRat v lo hi)
  benchLoop s!"dotIntervalUpper {labelBase}" iters (fun () =>
    Sound.Bounds.dotIntervalUpper v lo hi)
  benchLoop s!"dotIntervalUpperCommonDen {labelBase}" iters (fun () =>
    Sound.Bounds.dotIntervalUpperCommonDen v lo hi)
  benchLoop s!"dotIntervalUpperCachedRat {labelBase}" iters (fun () =>
    Sound.Bounds.dotIntervalUpperCachedRat v lo hi)

private def dotIntervalLowerCachedCore {n : Nat}
    (vArr loArr hiArr : Array Dyadic)
    (hv : vArr.size = n) (hlo : loArr.size = n) (hhi : hiArr.size = n) : Dyadic :=
  let term : Fin n → Dyadic := fun j =>
    let vj := vArr[j.1]'(by
      simp [hv, j.isLt])
    let loj := loArr[j.1]'(by
      simp [hlo, j.isLt])
    let hij := hiArr[j.1]'(by
      simp [hhi, j.isLt])
    if 0 ≤ vj then
      vj * loj
    else
      vj * hij
  Sound.Linear.sumFin n term

private def dotIntervalUpperCachedCore {n : Nat}
    (vArr loArr hiArr : Array Dyadic)
    (hv : vArr.size = n) (hlo : loArr.size = n) (hhi : hiArr.size = n) : Dyadic :=
  let term : Fin n → Dyadic := fun j =>
    let vj := vArr[j.1]'(by
      simp [hv, j.isLt])
    let loj := loArr[j.1]'(by
      simp [hlo, j.isLt])
    let hij := hiArr[j.1]'(by
      simp [hhi, j.isLt])
    if 0 ≤ vj then
      vj * hij
    else
      vj * loj
  Sound.Linear.sumFin n term

private def benchDotIntervalCachedParts (n iters seed : Nat) : IO Unit := do
  let (v, lo, hi) := mkInterval n seed
  let vArr := Array.ofFn v
  let loArr := Array.ofFn lo
  let hiArr := Array.ofFn hi
  have hv : vArr.size = n := by simp [vArr]
  have hlo : loArr.size = n := by simp [loArr]
  have hhi : hiArr.size = n := by simp [hiArr]
  let labelBase := s!"n={n}"
  benchLoop s!"dotIntervalLowerCachedRat arrays {labelBase}" iters (fun () =>
    let vArr' := Array.ofFn v
    let loArr' := Array.ofFn lo
    let hiArr' := Array.ofFn hi
    vArr'.size + loArr'.size + hiArr'.size)
  benchLoop s!"dotIntervalLowerCachedRat sum {labelBase}" iters (fun () =>
    dotIntervalLowerCachedCore vArr loArr hiArr hv hlo hhi)
  benchLoop s!"dotIntervalUpperCachedRat sum {labelBase}" iters (fun () =>
    dotIntervalUpperCachedCore vArr loArr hiArr hv hlo hhi)

private def benchDotFin (n iters seed : Nat) : IO Unit := do
  let x : Fin n → Dyadic := mkVecDyadic n seed 7 4
  let y : Fin n → Dyadic := mkVecDyadic n seed 19 5
  let labelBase := s!"n={n}"
  benchLoop s!"dotFin {labelBase}" iters (fun () =>
    Sound.Linear.dotFin n x y)

private def headShapeIters (base : Nat) : Nat :=
  max 1 (base / 10)

private def mkHeadAbs (seq dHead : Nat) (seed : Nat) (salt : Nat) : Fin seq → Fin dHead → Dyadic :=
  fun q d =>
    mkDyadic (q.1 * 31 + d.1 + seed + salt)
      (q.1 + d.1 + 7 + seed + salt) (((q.1 + d.1) % 3) = 0)

private def mkHeadVal (seq dHead : Nat) (seed : Nat) (salt : Nat) : Fin seq → Fin dHead → Dyadic :=
  fun q d =>
    mkDyadic (q.1 * 17 + d.1 + seed + salt)
      (q.1 + d.1 + 11 + seed + salt) (((q.1 + d.1) % 5) = 0)

private def mkHeadDir (dHead : Nat) (seed : Nat) (salt : Nat) : Fin dHead → Dyadic := fun d =>
  mkDyadic (d.1 + seed + salt) (d.1 + 3 + seed + salt) ((d.1 % 2) = 0)

private def benchHeadDotAbs (iters seed : Nat) : IO Unit := do
  let seq := 8
  let dHead := 64
  let qAbs : Fin seq → Fin dHead → Dyadic := mkHeadAbs seq dHead seed 3
  let kAbs : Fin seq → Fin dHead → Dyadic := mkHeadAbs seq dHead seed 19
  benchLoop "head dotAbs dotFin" iters (fun () =>
    (List.finRange seq).foldl (fun acc q =>
      (List.finRange seq).foldl (fun acc' k =>
        acc' + Sound.Linear.dotFin dHead (qAbs q) (kAbs k)) acc) 0)

private def benchHeadValueBounds (iters seed : Nat) : IO Unit := do
  let seq := 8
  let dHead := 64
  let dirHead : Fin dHead → Dyadic := mkHeadDir dHead seed 5
  let vLo : Fin seq → Fin dHead → Dyadic := mkHeadVal seq dHead seed 11
  let vHi : Fin seq → Fin dHead → Dyadic := mkHeadVal seq dHead seed 23
  let dirArr := Array.ofFn dirHead
  have hdir : dirArr.size = dHead := by simp [dirArr]
  let univ : Finset (Fin seq) := Finset.univ
  have hnonempty : univ.Nonempty := by
    simp [univ]
  benchLoop "head value bounds (cached)" iters (fun () =>
    let valsLo : Fin seq → Dyadic := fun k =>
      Sound.Bounds.dotIntervalLowerCachedRat dirHead (vLo k) (vHi k)
    let valsHi : Fin seq → Dyadic := fun k =>
      Sound.Bounds.dotIntervalUpperCachedRat dirHead (vLo k) (vHi k)
    let lo := univ.inf' hnonempty valsLo
    let hi := univ.sup' hnonempty valsHi
    lo + hi)
  benchLoop "head value bounds (common den)" iters (fun () =>
    let valsLo : Fin seq → Dyadic := fun k =>
      Sound.Bounds.dotIntervalLowerCommonDen dirHead (vLo k) (vHi k)
    let valsHi : Fin seq → Dyadic := fun k =>
      Sound.Bounds.dotIntervalUpperCommonDen dirHead (vLo k) (vHi k)
    let lo := univ.inf' hnonempty valsLo
    let hi := univ.sup' hnonempty valsHi
    lo + hi)
  benchLoop "head value bounds (direct)" iters (fun () =>
    let valsLo : Fin seq → Dyadic := fun k =>
      Sound.Bounds.dotIntervalLower dirHead (vLo k) (vHi k)
    let valsHi : Fin seq → Dyadic := fun k =>
      Sound.Bounds.dotIntervalUpper dirHead (vLo k) (vHi k)
    let lo := univ.inf' hnonempty valsLo
    let hi := univ.sup' hnonempty valsHi
    lo + hi)
  benchLoop "head value bounds (cached, reuse dir)" iters (fun () =>
    let valsLo : Fin seq → Dyadic := fun k =>
      let loArr := Array.ofFn (vLo k)
      let hiArr := Array.ofFn (vHi k)
      have hlo : loArr.size = dHead := by simp [loArr]
      have hhi : hiArr.size = dHead := by simp [hiArr]
      dotIntervalLowerCachedCore dirArr loArr hiArr hdir hlo hhi
    let valsHi : Fin seq → Dyadic := fun k =>
      let loArr := Array.ofFn (vLo k)
      let hiArr := Array.ofFn (vHi k)
      have hlo : loArr.size = dHead := by simp [loArr]
      have hhi : hiArr.size = dHead := by simp [hiArr]
      dotIntervalUpperCachedCore dirArr loArr hiArr hdir hlo hhi
    let lo := univ.inf' hnonempty valsLo
    let hi := univ.sup' hnonempty valsHi
    lo + hi)

private def benchDyadicDivInt (iters seed : Nat) : IO Unit := do
  let bigNum : Int :=
    Int.ofNat (2 ^ 200) * Int.ofNat (3 ^ 120) + Int.ofNat (5 ^ 90) + Int.ofNat seed
  let bigDen : Int :=
    Int.ofNat (2 ^ 150) * Int.ofNat (3 ^ 80) + (Int.ofNat seed) + 1
  benchLoop "dyadicOfRatDown divInt big" iters (fun () =>
    dyadicOfRatDown (Rat.divInt bigNum bigDen))

private def forceQkvSumLimited {seq dModel dHead : Nat}
    (qkv : Sound.HeadQKVBounds seq dModel dHead) (qLimit dLimit : Nat) : Dyadic :=
  let qs := (List.finRange seq).take qLimit
  let ds := (List.finRange dHead).take dLimit
  qs.foldl (fun acc q =>
    ds.foldl (fun acc' d =>
      acc' + qkv.qLo q d + qkv.qHi q d +
        qkv.kLo q d + qkv.kHi q d +
        qkv.vLo q d + qkv.vHi q d +
        qkv.qAbs q d + qkv.kAbs q d) acc) 0

private def forceQkvSumDirect {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (lnLo lnHi : Fin seq → Fin dModel → Dyadic)
    (qLimit dLimit : Nat)
    (dotLower dotUpper : (Fin dModel → Dyadic) → (Fin dModel → Dyadic) → (Fin dModel → Dyadic) → Dyadic) :
    Dyadic :=
  let qs := (List.finRange seq).take qLimit
  let ds := (List.finRange dHead).take dLimit
  qs.foldl (fun acc q =>
    ds.foldl (fun acc' d =>
      let qLo := dotLower (fun j => inputs.wq j d) (lnLo q) (lnHi q) + inputs.bq d
      let qHi := dotUpper (fun j => inputs.wq j d) (lnLo q) (lnHi q) + inputs.bq d
      let kLo := dotLower (fun j => inputs.wk j d) (lnLo q) (lnHi q) + inputs.bk d
      let kHi := dotUpper (fun j => inputs.wk j d) (lnLo q) (lnHi q) + inputs.bk d
      let vLo := dotLower (fun j => inputs.wv j d) (lnLo q) (lnHi q) + inputs.bv d
      let vHi := dotUpper (fun j => inputs.wv j d) (lnLo q) (lnHi q) + inputs.bv d
      let qAbs := max |qLo| |qHi|
      let kAbs := max |kLo| |kHi|
      acc' + qLo + qHi + kLo + kHi + vLo + vHi + qAbs + kAbs) acc) 0

private def benchHeadInputs {seq dModel dHead : Nat} [NeZero seq]
    (inputs : Model.InductionHeadInputs seq dModel dHead) (iters : Nat) : IO Unit := do
  IO.println "bench: head ln bounds start"
  (← IO.getStdout).flush
  let lnBounds ← Nfp.IO.timePure "bench: head ln bounds" (fun () =>
    Sound.headLnBounds inputs)
  IO.println "bench: head qkv bounds start"
  (← IO.getStdout).flush
  let qLimit :=
    match (← IO.getEnv "NFP_BENCH_QKV_Q") with
    | some raw => raw.toNat?.getD seq
    | none => seq
  let dLimit :=
    match (← IO.getEnv "NFP_BENCH_QKV_D") with
    | some raw => raw.toNat?.getD dHead
    | none => dHead
  let skipCache := (← IO.getEnv "NFP_BENCH_SKIP_QKV_CACHE").isSome
  if !skipCache then
    IO.println s!"bench: head qkv bounds (cachedDyadic) start q={qLimit} d={dLimit}"
    (← IO.getStdout).flush
    let _sumDyadic ← Nfp.IO.timePure "bench: head qkv bounds (cachedDyadic)" (fun () =>
      forceQkvSumLimited (Sound.headQKVBounds inputs lnBounds.1 lnBounds.2) qLimit dLimit)
    pure ()
  IO.println s!"bench: head qkv bounds (directDyadic) start q={qLimit} d={dLimit}"
  (← IO.getStdout).flush
  let _sumDirectDyadic ← Nfp.IO.timePure "bench: head qkv bounds (directDyadic)" (fun () =>
    forceQkvSumDirect inputs lnBounds.1 lnBounds.2 qLimit dLimit
      Sound.Bounds.dotIntervalLowerCachedRat Sound.Bounds.dotIntervalUpperCachedRat)
  IO.println s!"bench: head qkv bounds (directDyadicNoCache) start q={qLimit} d={dLimit}"
  (← IO.getStdout).flush
  let _sumDirectDyadicNoCache ← Nfp.IO.timePure "bench: head qkv bounds (directDyadicNoCache)" (fun () =>
    forceQkvSumDirect inputs lnBounds.1 lnBounds.2 qLimit dLimit
      Sound.Bounds.dotIntervalLower Sound.Bounds.dotIntervalUpper)
  let qkv := Sound.headQKVBounds inputs lnBounds.1 lnBounds.2
  let qAbs := qkv.qAbs
  let kAbs := qkv.kAbs
  benchLoop "head inputs dotAbs dotFin" iters (fun () =>
    (List.finRange seq).foldl (fun acc q =>
      (List.finRange seq).foldl (fun acc' k =>
        acc' + Sound.Linear.dotFin dHead (qAbs q) (kAbs k)) acc) 0)
  IO.println "bench: head value dir start"
  (← IO.getStdout).flush
  let dirHead ← Nfp.IO.timePure "bench: head value dir" (fun () =>
    Sound.headValueDirHead inputs)
  let dirArr := Array.ofFn dirHead
  have hdir : dirArr.size = dHead := by simp [dirArr]
  let univ : Finset (Fin seq) := Finset.univ
  have hnonempty : univ.Nonempty := by simp [univ]
  benchLoop "head inputs value bounds (cached)" iters (fun () =>
    let valsLo : Fin seq → Dyadic := fun k =>
      Sound.Bounds.dotIntervalLowerCachedRat dirHead (qkv.vLo k) (qkv.vHi k)
    let valsHi : Fin seq → Dyadic := fun k =>
      Sound.Bounds.dotIntervalUpperCachedRat dirHead (qkv.vLo k) (qkv.vHi k)
    let lo := univ.inf' hnonempty valsLo
    let hi := univ.sup' hnonempty valsHi
    lo + hi)
  benchLoop "head inputs value bounds (direct)" iters (fun () =>
    let valsLo : Fin seq → Dyadic := fun k =>
      Sound.Bounds.dotIntervalLower dirHead (qkv.vLo k) (qkv.vHi k)
    let valsHi : Fin seq → Dyadic := fun k =>
      Sound.Bounds.dotIntervalUpper dirHead (qkv.vLo k) (qkv.vHi k)
    let lo := univ.inf' hnonempty valsLo
    let hi := univ.sup' hnonempty valsHi
    lo + hi)
  benchLoop "head inputs value bounds (cached, reuse dir)" iters (fun () =>
    let valsLo : Fin seq → Dyadic := fun k =>
      let loArr := Array.ofFn (qkv.vLo k)
      let hiArr := Array.ofFn (qkv.vHi k)
      have hlo : loArr.size = dHead := by simp [loArr]
      have hhi : hiArr.size = dHead := by simp [hiArr]
      dotIntervalLowerCachedCore dirArr loArr hiArr hdir hlo hhi
    let valsHi : Fin seq → Dyadic := fun k =>
      let loArr := Array.ofFn (qkv.vLo k)
      let hiArr := Array.ofFn (qkv.vHi k)
      have hlo : loArr.size = dHead := by simp [loArr]
      have hhi : hiArr.size = dHead := by simp [hiArr]
      dotIntervalUpperCachedCore dirArr loArr hiArr hdir hlo hhi
    let lo := univ.inf' hnonempty valsLo
    let hi := univ.sup' hnonempty valsHi
    lo + hi)

/-- Run rational microbenchmarks for several vector sizes. -/
def runDyadicBench (seed : Nat) : IO Unit := do
  let baseIters :=
    match (← IO.getEnv "NFP_BENCH_ITERS") with
    | some raw => raw.toNat?.getD 200
    | none => 200
  let sizes : List Nat := [8, 64, 256, 768]
  for n in sizes do
    let iters := benchItersFor baseIters n
    IO.println s!"bench: start n={n} iters={iters}"
    benchDotInterval n iters seed
    benchDotIntervalCachedParts n iters seed
    benchDotFin n iters seed
  let headIters := headShapeIters baseIters
  IO.println s!"bench: start head-shape iters={headIters}"
  benchHeadDotAbs headIters seed
  benchHeadValueBounds headIters seed
  benchDyadicDivInt headIters seed

/-- Run benchmarks using a real induction-head input payload. -/
def runDyadicBenchFromInputs {seq dModel dHead : Nat} [NeZero seq]
    (seed : Nat) (inputs : Model.InductionHeadInputs seq dModel dHead) : IO Unit := do
  let skipSynth := (← IO.getEnv "NFP_BENCH_SKIP_SYNTH").isSome
  if !skipSynth then
    runDyadicBench seed
  let baseIters :=
    match (← IO.getEnv "NFP_BENCH_ITERS") with
    | some raw => raw.toNat?.getD 200
    | none => 200
  let headIters :=
    match (← IO.getEnv "NFP_BENCH_HEAD_ITERS") with
    | some raw => raw.toNat?.getD (headShapeIters baseIters)
    | none => headShapeIters baseIters
  IO.println s!"bench: start head-inputs iters={headIters}"
  (← IO.getStdout).flush
  benchHeadInputs inputs headIters

end IO

end Nfp
