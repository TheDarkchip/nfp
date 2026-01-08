-- SPDX-License-Identifier: AGPL-3.0-or-later

import Nfp.IO.Timing
import Nfp.Sound.Induction
import Nfp.Sound.Induction.HeadBounds

/-!
Benchmark helpers for induction-head core certification.
-/

namespace Nfp

namespace IO

open Sound
open scoped BigOperators

private def benchPhasePure {α : Type} (label : String) (act : Unit → α) : IO α := do
  IO.println s!"bench: {label} start"
  flushStdout
  timePhase label (pure (act ()))

private def forceScore {seq dModel dHead : Nat}
    (score : Sound.HeadScoreBounds seq dModel dHead) : Rat :=
  score.margin + score.eps

private def forceValues {seq dModel dHead : Nat}
    (vals : Sound.HeadValueBounds seq dModel dHead) : Rat :=
  vals.lo + vals.hi

private def forceQAbs {seq dHead : Nat}
    (qAbs : Fin seq → Fin dHead → Rat) : Rat :=
  (Finset.univ : Finset (Fin seq)).sum (fun q =>
    (Finset.univ : Finset (Fin dHead)).sum (fun d => qAbs q d))

private def forceLn {seq dModel : Nat}
    (ln : Fin seq → Fin dModel → Rat) : Rat :=
  (Finset.univ : Finset (Fin seq)).sum (fun q =>
    (Finset.univ : Finset (Fin dModel)).sum (fun i => ln q i))

private def forceKAbs {seq dHead : Nat}
    (kAbs : Fin seq → Fin dHead → Rat) : Rat :=
  (Finset.univ : Finset (Fin seq)).sum (fun q =>
    (Finset.univ : Finset (Fin dHead)).sum (fun d => kAbs q d))

private def forceDotAbs {seq dHead : Nat}
    (qAbs kAbs : Fin seq → Fin dHead → Rat) : Rat :=
  (Finset.univ : Finset (Fin seq)).sum (fun q =>
    (Finset.univ : Finset (Fin seq)).sum (fun k =>
      Linear.dotFin dHead (fun d => qAbs q d) (fun d => kAbs k d)))

private def forceDotAbsTasksReduce {seq dHead : Nat}
    (qAbs kAbs : Fin seq → Fin dHead → Rat) : Rat :=
  let tasks : Array (Task Rat) :=
    Array.ofFn (fun q : Fin seq =>
      Task.spawn (fun _ =>
        (Finset.univ : Finset (Fin seq)).sum (fun k =>
          Linear.dotFin dHead (fun d => qAbs q d) (fun d => kAbs k d))))
  (Finset.univ : Finset (Fin seq)).sum (fun q =>
    (tasks[q.1]'(by simp [tasks, q.isLt])).get)

private def isPow2 (n : Nat) : Bool :=
  if n = 0 then
    false
  else
    decide (Nat.pow 2 (Nat.log2 n) = n)

private def isPow2Den (q : Rat) : Bool :=
  isPow2 q.den

private def countPow2Den {seq dHead : Nat}
    (qs : List (Fin seq)) (ds : List (Fin dHead))
    (f : Fin seq → Fin dHead → Rat) : Nat :=
  qs.foldl (fun acc q =>
    ds.foldl (fun acc' d => acc' + (if isPow2Den (f q d) then 1 else 0)) acc) 0

private def pow2DenSampleReport {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (qkv : Sound.HeadQKVBounds seq dModel dHead) : String :=
  let qs := (List.finRange seq).take (min seq 2)
  let ds := (List.finRange dHead).take (min dHead 8)
  let total := qs.length * ds.length
  let qLoDy := countPow2Den qs ds qkv.qLo
  let qHiDy := countPow2Den qs ds qkv.qHi
  let qAbsDy := countPow2Den qs ds qkv.qAbs
  let kAbsDy := countPow2Den qs ds qkv.kAbs
  let epsDy := if isPow2Den inputs.lnEps then 1 else 0
  s!"pow2-den sample: total={total} qLo={qLoDy} qHi={qHiDy} qAbs={qAbsDy} " ++
    s!"kAbs={kAbsDy} lnEps={epsDy}"

private def pow2DenSanityReport : String :=
  let rat := ratRoundDown (Rat.divInt 1 8)
  let powChecks :=
    s!"pow2(1)={isPow2 1} pow2(2)={isPow2 2} pow2(3)={isPow2 3} " ++
      s!"pow2(4)={isPow2 4} pow2(8)={isPow2 8}"
  let ratCheck := s!"rat(1/8).den={rat.den} pow2den={isPow2Den rat}"
  s!"pow2-den sanity: {powChecks} {ratCheck}"

private def forceQRowTasks {seq dHead : Nat}
    (q0 : Fin seq) (qLo : Fin seq → Fin dHead → Rat) : Int :=
  let tasks : Array (Task Rat) :=
    Array.ofFn (fun d : Fin dHead =>
      Task.spawn (fun _ => qLo q0 d))
  let total :=
    (Finset.univ : Finset (Fin dHead)).sum (fun d =>
      (tasks[d.1]'(by simp [tasks, d.isLt])).get)
  total.num

private def qAbsRowChunk {seq dHead : Nat}
    (q0 : Fin seq) (qAbs : Fin seq → Fin dHead → Rat) (start stop : Nat) : Rat :=
  let chunk : Finset (Fin dHead) :=
    (Finset.univ : Finset (Fin dHead)).filter (fun d => start ≤ d.1 ∧ d.1 < stop)
  chunk.sum (fun d => qAbs q0 d)

private def forceQAbsRowTasksReduce {seq dHead : Nat}
    (q0 : Fin seq) (qAbs : Fin seq → Fin dHead → Rat) (chunkSize : Nat) : Rat :=
  if chunkSize = 0 then
    (0 : Rat)
  else
    let chunks : Nat := (dHead + chunkSize - 1) / chunkSize
    let tasks : Array (Task Rat) :=
      Array.ofFn (fun i : Fin chunks =>
        Task.spawn (fun _ =>
          let start := i.1 * chunkSize
          let stop := min dHead (start + chunkSize)
          qAbsRowChunk q0 qAbs start stop))
    (Finset.univ : Finset (Fin chunks)).sum (fun i =>
      (tasks[i.1]'(by simp [tasks, i.isLt])).get)

private def forceQAbsAllTasksReduce {seq dHead : Nat}
    (qAbs : Fin seq → Fin dHead → Rat) (chunkSize : Nat) : Rat :=
  let tasks : Array (Task Rat) :=
    Array.ofFn (fun q : Fin seq =>
      Task.spawn (fun _ => forceQAbsRowTasksReduce q qAbs chunkSize))
  (Finset.univ : Finset (Fin seq)).sum (fun q =>
    (tasks[q.1]'(by simp [tasks, q.isLt])).get)

private def forceQAbsActiveTasksReduce {seq dHead : Nat}
    (active : Finset (Fin seq)) (qAbs : Fin seq → Fin dHead → Rat) (chunkSize : Nat) : Rat :=
  if hactive : active.Nonempty then
    let tasks : Array (Task Rat) :=
      Array.ofFn (fun q : Fin seq =>
        Task.spawn (fun _ =>
          if q ∈ active then
            forceQAbsRowTasksReduce q qAbs chunkSize
          else
            (0 : Rat)))
    active.sum (fun q =>
      (tasks[q.1]'(by simp [tasks, q.isLt])).get)
  else
    (0 : Rat)

/-- Run a core benchmark from already-parsed head inputs. -/
def runCoreBenchFromInputs {seq dModel dHead : Nat} [NeZero seq]
    (inputs : Model.InductionHeadInputs seq dModel dHead) : IO Unit := do
  let lnBounds ← benchPhasePure "ln bounds" (fun () => Sound.headLnBounds inputs)
  let lnLo := lnBounds.1
  let lnHi := lnBounds.2
  let _ ← benchPhasePure "lnLo force" (fun () => forceLn lnLo)
  let _ ← benchPhasePure "lnHi force" (fun () => forceLn lnHi)
  let qkv ← benchPhasePure "qkv bounds" (fun () => Sound.headQKVBounds inputs lnLo lnHi)
  let _ ← timePhase "pow2-den sample" (do
    IO.println (pow2DenSampleReport inputs qkv)
    IO.println pow2DenSanityReport
    pure ())
  let _ ← benchPhasePure "qLo single" (fun () =>
    match h : dHead with
    | 0 => (0 : Rat)
    | Nat.succ _ =>
        let q0 : Fin seq :=
          ⟨0, Nat.pos_of_ne_zero (NeZero.ne (n := seq))⟩
        let d0 : Fin dHead := ⟨0, by simp [h]⟩
        qkv.qLo q0 d0)
  let _ ← benchPhasePure "qLo row tasks" (fun () =>
    let q0 : Fin seq :=
      ⟨0, by
        exact Nat.pos_of_ne_zero (NeZero.ne (n := seq))⟩
    forceQRowTasks q0 qkv.qLo)
  let _ ← benchPhasePure "qLo row" (fun () =>
    let q0 : Fin seq :=
      ⟨0, by
        exact Nat.pos_of_ne_zero (NeZero.ne (n := seq))⟩
    let total :=
      (Finset.univ : Finset (Fin dHead)).sum (fun d => qkv.qLo q0 d)
    total.num)
  let _ ← benchPhasePure "qLo force" (fun () => forceQAbs qkv.qLo)
  let _ ← benchPhasePure "qHi force" (fun () => forceQAbs qkv.qHi)
  let _ ← benchPhasePure "qAbs single" (fun () =>
    let q0 : Fin seq :=
      ⟨0, by
        exact Nat.pos_of_ne_zero (NeZero.ne (n := seq))⟩
    match h : dHead with
    | 0 => (0 : Rat)
    | Nat.succ _ =>
        let d0 : Fin dHead := ⟨0, by simp [h]⟩
        qkv.qAbs q0 d0)
  let _ ← benchPhasePure "qAbs row tasks reduce" (fun () =>
    let q0 : Fin seq :=
      ⟨0, by
        exact Nat.pos_of_ne_zero (NeZero.ne (n := seq))⟩
    forceQAbsRowTasksReduce q0 qkv.qAbs 1)
  let _ ← benchPhasePure "qAbs force tasks reduce active" (fun () =>
    forceQAbsActiveTasksReduce inputs.active qkv.qAbs 1)
  let _ ← benchPhasePure "qAbs force tasks reduce" (fun () =>
    forceQAbsAllTasksReduce qkv.qAbs 1)
  let _ ← benchPhasePure "kAbs force tasks reduce active" (fun () =>
    forceQAbsActiveTasksReduce inputs.active qkv.kAbs 1)
  let _ ← benchPhasePure "kAbs force tasks reduce" (fun () =>
    forceQAbsAllTasksReduce qkv.kAbs 1)
  let _ ← benchPhasePure "kAbs force tasks reduce (bench)" (fun () =>
    forceQAbsAllTasksReduce qkv.kAbs 1)
  let _ ← benchPhasePure "dotAbs force tasks reduce" (fun () =>
    forceDotAbsTasksReduce qkv.qAbs qkv.kAbs)
  let _ ← benchPhasePure "dotAbs force" (fun () => forceDotAbs qkv.qAbs qkv.kAbs)
  let score ← benchPhasePure "score bounds" (fun () =>
    Sound.headScoreBounds inputs qkv.qAbs qkv.kAbs)
  let _ ← benchPhasePure "score force" (fun () => forceScore score)
  let vals ← benchPhasePure "value bounds" (fun () =>
    Sound.headValueBounds inputs qkv.vLo qkv.vHi)
  let _ ← benchPhasePure "value force" (fun () => forceValues vals)
  let cert ← benchPhasePure "core cert" (fun () =>
    Sound.buildInductionCertFromHeadCore? inputs)
  match cert with
  | none => IO.println "bench: core cert none"
  | some _ => IO.println "bench: core cert some"

end IO

end Nfp
