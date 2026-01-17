-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Core.Basic

/-!
Caching helpers for interval bounds.
-/

public section

namespace Nfp


namespace Bounds

/-- Cache a bound function in an array-backed lookup to avoid repeated evaluation. -/
def cacheBound {n : Nat} (f : Fin n → Rat) : Fin n → Rat :=
  let data : Thunk (Array Rat) := Thunk.mk (fun _ => Array.ofFn f)
  fun i => (Thunk.get data)[i.1]'(by
    simp [Thunk.get, data, i.isLt])

/-- `cacheBound` preserves pointwise values. -/
theorem cacheBound_apply {n : Nat} (f : Fin n → Rat) (i : Fin n) :
    cacheBound f i = f i := by
  simp [cacheBound, Thunk.get, Array.getElem_ofFn]

/-- Cache a bound function using per-index thunks for lazy evaluation. -/
def cacheBoundThunk {n : Nat} (f : Fin n → Rat) : Fin n → Rat :=
  let data : Array (Thunk Rat) := Array.ofFn (fun i => Thunk.mk (fun _ => f i))
  fun i =>
    let t := data[i.1]'(by
      simp [data, i.isLt])
    Thunk.get t

/-- `cacheBoundThunk` preserves pointwise values. -/
theorem cacheBoundThunk_apply {n : Nat} (f : Fin n → Rat) (i : Fin n) :
    cacheBoundThunk f i = f i := by
  simp [cacheBoundThunk, Thunk.get, Array.getElem_ofFn]

/-- Cache a bound function using tasks for parallel evaluation. -/
def cacheBoundTask {n : Nat} (f : Fin n → Rat) : Fin n → Rat :=
  let tasks : Array (Task Rat) :=
    Array.ofFn (fun i : Fin n =>
      Task.spawn (fun _ => f i))
  fun i =>
    let t := tasks[i.1]'(by
      simp [tasks, i.isLt])
    t.get

/-- `cacheBoundTask` preserves pointwise values. -/
theorem cacheBoundTask_apply {n : Nat} (f : Fin n → Rat) (i : Fin n) :
    cacheBoundTask f i = f i := by
  classical
  simp [cacheBoundTask, Task.spawn, Array.getElem_ofFn]

/-- Cache a bound function on two indices. -/
def cacheBound2 {m n : Nat} (f : Fin m → Fin n → Rat) : Fin m → Fin n → Rat :=
  let data : Thunk (Array (Thunk (Array Rat))) := Thunk.mk (fun _ =>
    Array.ofFn (fun q => Thunk.mk (fun _ => Array.ofFn (f q))))
  fun q i =>
    let rows := Thunk.get data
    let rowThunk := rows[q.1]'(by
      simp [rows, Thunk.get, data, q.isLt])
    let row := Thunk.get rowThunk
    row[i.1]'(by
      simp [row, rowThunk, rows, Thunk.get, data, i.isLt])

/-- `cacheBound2` preserves pointwise values. -/
theorem cacheBound2_apply {m n : Nat} (f : Fin m → Fin n → Rat) (q : Fin m) (i : Fin n) :
    cacheBound2 f q i = f q i := by
  simp [cacheBound2, Thunk.get, Array.getElem_ofFn]

/-- Cache a bound function on two indices using row tasks for parallel evaluation. -/
def cacheBound2Task {m n : Nat} (f : Fin m → Fin n → Rat) : Fin m → Fin n → Rat :=
  let rowTasks : Array (Task { row : Array Rat // row.size = n }) :=
    Array.ofFn (fun q : Fin m =>
      Task.spawn (fun _ => ⟨Array.ofFn (f q), by simp⟩))
  fun q i =>
    let row := (rowTasks[q.1]'(by
      simp [rowTasks, q.isLt])).get
    row.1[i.1]'(by
      simp [row.2])

/-- `cacheBound2Task` preserves pointwise values. -/
theorem cacheBound2Task_apply {m n : Nat} (f : Fin m → Fin n → Rat) (q : Fin m) (i : Fin n) :
    cacheBound2Task f q i = f q i := by
  classical
  simp [cacheBound2Task, Task.spawn, Array.getElem_ofFn]

/-- Cache a bound function on two indices using per-element tasks for parallel evaluation. -/
def cacheBound2TaskElem {m n : Nat} (f : Fin m → Fin n → Rat) : Fin m → Fin n → Rat :=
  let rowTasks : Array (Array (Task Rat)) :=
    Array.ofFn (fun q : Fin m =>
      Array.ofFn (fun i : Fin n =>
        Task.spawn (fun _ => f q i)))
  fun q i =>
    let row := (rowTasks[q.1]'(by
      simp [rowTasks, q.isLt]))
    let t := row[i.1]'(by
      simp [row, rowTasks, i.isLt])
    t.get

/-- `cacheBound2TaskElem` preserves pointwise values. -/
theorem cacheBound2TaskElem_apply {m n : Nat} (f : Fin m → Fin n → Rat) (q : Fin m) (i : Fin n) :
    cacheBound2TaskElem f q i = f q i := by
  classical
  simp [cacheBound2TaskElem, Task.spawn, Array.getElem_ofFn]

/-- Cache a pair of bound functions on two indices. -/
def cacheBoundPair2 {m n : Nat}
    (f : Fin m → (Fin n → Rat) × (Fin n → Rat)) :
    (Fin m → Fin n → Rat) × (Fin m → Fin n → Rat) :=
  let data : Thunk (Array (Array Rat × Array Rat)) := Thunk.mk (fun _ =>
    Array.ofFn (fun q =>
      let row := f q
      (Array.ofFn row.1, Array.ofFn row.2)))
  let lo : Fin m → Fin n → Rat := fun q i =>
    let rows := Thunk.get data
    let row := rows[q.1]'(by
      simp [rows, Thunk.get, data, q.isLt])
    let loRow := row.1
    loRow[i.1]'(by
      simp [loRow, row, rows, Thunk.get, data, i.isLt])
  let hi : Fin m → Fin n → Rat := fun q i =>
    let rows := Thunk.get data
    let row := rows[q.1]'(by
      simp [rows, Thunk.get, data, q.isLt])
    let hiRow := row.2
    hiRow[i.1]'(by
      simp [hiRow, row, rows, Thunk.get, data, i.isLt])
  (lo, hi)

/-- `cacheBoundPair2` preserves pointwise values (first component). -/
theorem cacheBoundPair2_apply_left {m n : Nat}
    (f : Fin m → (Fin n → Rat) × (Fin n → Rat)) (q : Fin m) (i : Fin n) :
    (cacheBoundPair2 f).1 q i = (f q).1 i := by
  simp [cacheBoundPair2, Thunk.get, Array.getElem_ofFn]

/-- `cacheBoundPair2` preserves pointwise values (second component). -/
theorem cacheBoundPair2_apply_right {m n : Nat}
    (f : Fin m → (Fin n → Rat) × (Fin n → Rat)) (q : Fin m) (i : Fin n) :
    (cacheBoundPair2 f).2 q i = (f q).2 i := by
  simp [cacheBoundPair2, Thunk.get, Array.getElem_ofFn]

/-- Cache a pair of bound functions on two indices using row tasks. -/
def cacheBoundPair2Task {m n : Nat}
    (f : Fin m → (Fin n → Rat) × (Fin n → Rat)) :
    (Fin m → Fin n → Rat) × (Fin m → Fin n → Rat) :=
  let rowTasks :
      Array (Task ({ rowLo : Array Rat // rowLo.size = n } ×
        { rowHi : Array Rat // rowHi.size = n })) :=
    Array.ofFn (fun q : Fin m =>
      Task.spawn (fun _ =>
        let row := f q
        (⟨Array.ofFn row.1, by simp⟩, ⟨Array.ofFn row.2, by simp⟩)))
  let lo : Fin m → Fin n → Rat := fun q i =>
    let row := (rowTasks[q.1]'(by
      simp [rowTasks, q.isLt])).get
    row.1.1[i.1]'(by
      simp [row.1.2, i.isLt])
  let hi : Fin m → Fin n → Rat := fun q i =>
    let row := (rowTasks[q.1]'(by
      simp [rowTasks, q.isLt])).get
    row.2.1[i.1]'(by
      simp [row.2.2, i.isLt])
  (lo, hi)

/-- `cacheBoundPair2Task` preserves pointwise values (first component). -/
theorem cacheBoundPair2Task_apply_left {m n : Nat}
    (f : Fin m → (Fin n → Rat) × (Fin n → Rat)) (q : Fin m) (i : Fin n) :
    (cacheBoundPair2Task f).1 q i = (f q).1 i := by
  classical
  simp [cacheBoundPair2Task, Task.spawn, Array.getElem_ofFn]

/-- `cacheBoundPair2Task` preserves pointwise values (second component). -/
theorem cacheBoundPair2Task_apply_right {m n : Nat}
    (f : Fin m → (Fin n → Rat) × (Fin n → Rat)) (q : Fin m) (i : Fin n) :
    (cacheBoundPair2Task f).2 q i = (f q).2 i := by
  classical
  simp [cacheBoundPair2Task, Task.spawn, Array.getElem_ofFn]

/-- Cache a pair of bound functions on two indices using per-element tasks. -/
def cacheBoundPair2TaskElem {m n : Nat} (f : Fin m → Fin n → Rat × Rat) :
    (Fin m → Fin n → Rat) × (Fin m → Fin n → Rat) :=
  let rowTasks : Array (Array (Task (Rat × Rat))) :=
    Array.ofFn (fun q : Fin m =>
      Array.ofFn (fun i : Fin n =>
        Task.spawn (fun _ => f q i)))
  let lo : Fin m → Fin n → Rat := fun q i =>
    let row := (rowTasks[q.1]'(by
      simp [rowTasks, q.isLt]))
    let t := row[i.1]'(by
      simp [row, rowTasks, i.isLt])
    (t.get).1
  let hi : Fin m → Fin n → Rat := fun q i =>
    let row := (rowTasks[q.1]'(by
      simp [rowTasks, q.isLt]))
    let t := row[i.1]'(by
      simp [row, rowTasks, i.isLt])
    (t.get).2
  (lo, hi)

/-- `cacheBoundPair2TaskElem` preserves pointwise values (first component). -/
theorem cacheBoundPair2TaskElem_apply_left {m n : Nat}
    (f : Fin m → Fin n → Rat × Rat) (q : Fin m) (i : Fin n) :
    (cacheBoundPair2TaskElem f).1 q i = (f q i).1 := by
  classical
  simp [cacheBoundPair2TaskElem, Task.spawn, Array.getElem_ofFn]

/-- `cacheBoundPair2TaskElem` preserves pointwise values (second component). -/
theorem cacheBoundPair2TaskElem_apply_right {m n : Nat}
    (f : Fin m → Fin n → Rat × Rat) (q : Fin m) (i : Fin n) :
    (cacheBoundPair2TaskElem f).2 q i = (f q i).2 := by
  classical
  simp [cacheBoundPair2TaskElem, Task.spawn, Array.getElem_ofFn]

end Bounds


end Nfp
