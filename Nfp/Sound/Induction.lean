-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Sound.Induction.Core
public import Nfp.Sound.Induction.HeadOutput
public import Nfp.Sound.Induction.LogitDiff
public import Nfp.Sound.Induction.OneHot
public import Nfp.Sound.Induction.ScoreBounds
public import Nfp.Sound.Induction.ScoreIntervals
public import Nfp.Sound.Induction.TransformerStack
public import Nfp.Sound.Induction.ValueBounds

/-!
Soundness lemmas for induction certificates.

This module re-exports the core definitions, head-output helpers, and logit-diff
lemmas that operate on explicit certificates.
-/
