-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Core.Basic
public import Nfp.Sound.Linear.FinFold

/-!
Pure helpers for building cached dot-abs functions for head scoring.
-/

public section

namespace Nfp

namespace IO

/-- Build a cached dot-abs function from Q/K absolute bounds using tasks. -/
def dotAbsFromQKV {seq dHead : Nat}
    (qAbs kAbs : Fin seq → Fin dHead → Rat) : Fin seq → Fin seq → Rat :=
  let rowTasks : Array (Task (Array Rat)) :=
    Array.ofFn (fun q : Fin seq =>
      Task.spawn (fun _ =>
        Array.ofFn (fun k : Fin seq =>
          Sound.Linear.dotFin dHead (fun d => qAbs q d) (fun d => kAbs k d))))
  let cache : Array (Array Rat) :=
    Array.ofFn (fun q : Fin seq =>
      (rowTasks[q.1]'(by
        simp [rowTasks, q.isLt])).get)
  fun q k =>
    let row := cache[q.1]'(by
      simp [cache, q.isLt])
    row[k.1]'(by
      have hrow : row.size = seq := by
        simp [row, cache, rowTasks, Task.spawn]
      simp [hrow, k.isLt])

private theorem dotAbsFromQKV_spec {seq dHead : Nat}
    (qAbs kAbs : Fin seq → Fin dHead → Rat) :
    dotAbsFromQKV qAbs kAbs =
      let rowTasks : Array (Task (Array Rat)) :=
        Array.ofFn (fun q : Fin seq =>
          Task.spawn (fun _ =>
            Array.ofFn (fun k : Fin seq =>
              Sound.Linear.dotFin dHead (fun d => qAbs q d) (fun d => kAbs k d))))
      let cache : Array (Array Rat) :=
        Array.ofFn (fun q : Fin seq =>
          (rowTasks[q.1]'(by
            simp [rowTasks, q.isLt])).get)
      fun q k =>
        let row := cache[q.1]'(by
          simp [cache, q.isLt])
        row[k.1]'(by
          have hrow : row.size = seq := by
            simp [row, cache, rowTasks, Task.spawn]
          simp [hrow, k.isLt]) := rfl

end IO

end Nfp
