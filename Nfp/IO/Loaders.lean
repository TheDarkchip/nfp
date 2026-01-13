-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.IO.Pure
public import Nfp.Circuit.Cert.LogitDiff
public import Nfp.Circuit.Cert.DownstreamLinear
public import Nfp.Circuit.Cert.ResidualBound
public import Nfp.Circuit.Cert.ResidualInterval

/-!
IO loaders for certificates and raw inputs.
-/

public section

namespace Nfp

namespace IO

open Nfp.Circuit

/-- Load a softmax-margin certificate from disk. -/
def loadSoftmaxMarginCert (path : System.FilePath) :
    IO (Except String (Sigma SoftmaxMarginCert)) := do
  let data ← IO.FS.readFile path
  return Pure.parseSoftmaxMarginCert data

/-- Load raw softmax-margin inputs from disk. -/
def loadSoftmaxMarginRaw (path : System.FilePath) :
    IO (Except String (Sigma Pure.SoftmaxMarginRaw)) := do
  let data ← IO.FS.readFile path
  return Pure.parseSoftmaxMarginRaw data

/-- Load a value-range certificate from disk. -/
def loadValueRangeCert (path : System.FilePath) :
    IO (Except String (Sigma ValueRangeCert)) := do
  let data ← IO.FS.readFile path
  return Pure.parseValueRangeCert data

/-- Load a downstream linear certificate from disk. -/
def loadDownstreamLinearCert (path : System.FilePath) :
    IO (Except String DownstreamLinearCert) := do
  let data ← IO.FS.readFile path
  return Pure.parseDownstreamLinearCert data

/-- Load a downstream matrix payload from disk. -/
def loadDownstreamMatrixRaw (path : System.FilePath) :
    IO (Except String (Sigma (fun rows =>
      Sigma (fun cols => Pure.DownstreamMatrixRaw rows cols)))) := do
  let data ← IO.FS.readFile path
  return Pure.parseDownstreamMatrixRaw data

/-- Load a residual-bound certificate from disk. -/
def loadResidualBoundCert (path : System.FilePath) :
    IO (Except String (Sigma (fun n => ResidualBoundCert n))) := do
  let data ← IO.FS.readFile path
  return Pure.parseResidualBoundCert data

/-- Load a residual-interval certificate from disk. -/
def loadResidualIntervalCert (path : System.FilePath) :
    IO (Except String (Sigma (fun n => ResidualIntervalCert n))) := do
  let data ← IO.FS.readFile path
  return Pure.parseResidualIntervalCert data

/-- Load raw value-range inputs from disk. -/
def loadValueRangeRaw (path : System.FilePath) :
    IO (Except String (Sigma Pure.ValueRangeRaw)) := do
  let data ← IO.FS.readFile path
  return Pure.parseValueRangeRaw data

end IO

end Nfp
