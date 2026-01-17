-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.IO.Parse
public import Nfp.Circuit.Cert.DownstreamLinear
public import Nfp.Circuit.Cert.ResidualBound
public import Nfp.Circuit.Cert.ResidualInterval

/-!
IO loaders for certificates.
-/

public section

namespace Nfp

namespace IO

open Nfp.Circuit

/-- Load a softmax-margin certificate from disk. -/
def loadSoftmaxMarginCert (path : System.FilePath) :
    IO (Except String (Sigma SoftmaxMarginCert)) := do
  let data ← IO.FS.readFile path
  return Parse.parseSoftmaxMarginCert data

/-- Load a value-range certificate from disk. -/
def loadValueRangeCert (path : System.FilePath) :
    IO (Except String (Sigma ValueRangeCert)) := do
  let data ← IO.FS.readFile path
  return Parse.parseValueRangeCert data

/-- Load a downstream linear certificate from disk. -/
def loadDownstreamLinearCert (path : System.FilePath) :
    IO (Except String DownstreamLinearCert) := do
  let data ← IO.FS.readFile path
  return Parse.parseDownstreamLinearCert data

/-- Load a residual-bound certificate from disk. -/
def loadResidualBoundCert (path : System.FilePath) :
    IO (Except String (Sigma (fun n => ResidualBoundCert n))) := do
  let data ← IO.FS.readFile path
  return Parse.parseResidualBoundCert data

/-- Load a residual-interval certificate from disk. -/
def loadResidualIntervalCert (path : System.FilePath) :
    IO (Except String (Sigma (fun n => ResidualIntervalCert n))) := do
  let data ← IO.FS.readFile path
  return Parse.parseResidualIntervalCert data

end IO

end Nfp
