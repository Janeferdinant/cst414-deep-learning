import React, { useState } from 'react';
import { 
  Layers, Cpu, Play, RefreshCw, Activity, 
  HelpCircle, ChevronRight, Sliders, Eye
} from 'lucide-react';

export const CnnRnnStudio: React.FC = () => {
  const [activeSubTab, setActiveSubTab] = useState<'cnn' | 'lstm'>('cnn');

  // --- CNN STATE ---
  const [kernelType, setKernelType] = useState<'sobelH' | 'sobelV' | 'sharpen' | 'edge' | 'blur'>('edge');
  const [stride, setStride] = useState<number>(1);
  const [padding, setPadding] = useState<number>(0);
  const [selectedCell, setSelectedCell] = useState<{ r: number; c: number }>({ r: 0, c: 0 });

  // CNN Dimension calculator state
  const [calcWin, setCalcWin] = useState<number>(224);
  const [calcK, setCalcK] = useState<number>(3);
  const [calcP, setCalcP] = useState<number>(1);
  const [calcS, setCalcS] = useState<number>(1);

  // 5x5 Input Image Sample (Values 0 to 9)
  const inputMatrix = [
    [1, 2, 0, 1, 3],
    [0, 1, 3, 2, 0],
    [3, 0, 1, 4, 2],
    [2, 1, 0, 2, 1],
    [0, 3, 2, 1, 0]
  ];

  // Kernels
  const kernels = {
    edge: [
      [-1, -1, -1],
      [-1,  8, -1],
      [-1, -1, -1]
    ],
    sobelH: [
      [-1, -2, -1],
      [ 0,  0,  0],
      [ 1,  2,  1]
    ],
    sobelV: [
      [-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1]
    ],
    sharpen: [
      [ 0, -1,  0],
      [-1,  5, -1],
      [ 0, -1,  0]
    ],
    blur: [
      [1, 2, 1],
      [2, 4, 2],
      [1, 2, 1]
    ]
  };

  const currentKernel = kernels[kernelType];

  // Convolution computation for 5x5 input and 3x3 kernel (valid padding, stride 1)
  const computeFeatureMap = () => {
    const kSize = 3;
    const outH = Math.floor((5 - kSize + 2 * padding) / stride) + 1;
    const outW = Math.floor((5 - kSize + 2 * padding) / stride) + 1;
    const res: number[][] = [];

    for (let r = 0; r < outH; r++) {
      const row: number[] = [];
      for (let c = 0; c < outW; c++) {
        let sum = 0;
        for (let i = 0; i < kSize; i++) {
          for (let j = 0; j < kSize; j++) {
            const inR = r * stride + i - padding;
            const inC = c * stride + j - padding;
            const val = (inR >= 0 && inR < 5 && inC >= 0 && inC < 5) ? inputMatrix[inR][inC] : 0;
            sum += val * currentKernel[i][j];
          }
        }
        row.push(sum);
      }
      res.push(row);
    }
    return res;
  };

  const featureMap = computeFeatureMap();

  // Current selected patch calculation
  const getSelectedPatchDetails = () => {
    const r = selectedCell.r;
    const c = selectedCell.c;
    const kSize = 3;
    const terms: string[] = [];
    let total = 0;

    for (let i = 0; i < kSize; i++) {
      for (let j = 0; j < kSize; j++) {
        const inR = r * stride + i - padding;
        const inC = c * stride + j - padding;
        const val = (inR >= 0 && inR < 5 && inC >= 0 && inC < 5) ? inputMatrix[inR][inC] : 0;
        const weight = currentKernel[i][j];
        const prod = val * weight;
        total += prod;
        terms.push(`(${val} × ${weight})`);
      }
    }
    return { terms, total, r, c };
  };

  const patchDetails = getSelectedPatchDetails();

  // Pooling Demo Matrix (4x4)
  const poolMatrix = [
    [12, 20, 30,  0],
    [ 8, 12,  2, 14],
    [34, 70, 37,  4],
    [12,  1, 45, 23]
  ];

  const maxPool2x2 = [
    [Math.max(12, 20, 8, 12), Math.max(30, 0, 2, 14)],
    [Math.max(34, 70, 12, 1), Math.max(37, 4, 45, 23)]
  ];

  const avgPool2x2 = [
    [Math.round((12 + 20 + 8 + 12) / 4), Math.round((30 + 0 + 2 + 14) / 4)],
    [Math.round((34 + 70 + 12 + 1) / 4), Math.round((37 + 4 + 45 + 23) / 4)]
  ];

  // Output dimension calculation
  const calculatedOutputDim = Math.floor((calcWin - calcK + 2 * calcP) / calcS) + 1;

  // --- LSTM STATE ---
  const [cPrev, setCPrev] = useState<number>(0.8);
  const [hPrev, setHPrev] = useState<number>(0.4);
  const [forgetGate, setForgetGate] = useState<number>(0.85); // f_t in [0, 1]
  const [inputGate, setInputGate] = useState<number>(0.65);  // i_t in [0, 1]
  const [candC, setCandC] = useState<number>(0.50);          // c~_t in [-1, 1]
  const [outputGate, setOutputGate] = useState<number>(0.75);// o_t in [0, 1]

  // LSTM Formulas
  // C_t = f_t * C_{t-1} + i_t * c~_t
  const cCur = forgetGate * cPrev + inputGate * candC;
  // h_t = o_t * tanh(C_t)
  const hCur = outputGate * Math.tanh(cCur);

  return (
    <div className="space-y-6">
      
      {/* Studio Header */}
      <div className="p-6 rounded-2xl bg-gradient-to-r from-navy-900 via-burgundy-950/60 to-navy-900 border border-gold-500/30 flex flex-col md:flex-row items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-burgundy-800 text-gold-300 border border-gold-500/30">
              Interactive Architecture Lab
            </span>
            <span className="text-xs text-slate-400 font-medium">Modules 3 &amp; 4 Core Concepts</span>
          </div>
          <h2 className="text-lg md:text-xl font-bold font-serif text-slate-100 mt-1">
            CNN Spatial Feature Extraction &amp; LSTM Gated Recurrent Dataflow
          </h2>
          <p className="text-xs text-slate-300 mt-1">
            Simulate 2D cross-correlation, kernel transformations, pooling invariance, and LSTM cell state preservation.
          </p>
        </div>

        {/* Tab Switcher */}
        <div className="flex items-center bg-navy-950 rounded-xl p-1 border border-gold-500/30 shadow-lg">
          <button
            onClick={() => setActiveSubTab('cnn')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-bold transition-all ${
              activeSubTab === 'cnn'
                ? 'bg-burgundy-800 text-gold-200 border border-gold-500/40 shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Layers className="w-4 h-4 text-gold-400" />
            <span>Convolution &amp; Pooling Studio</span>
          </button>

          <button
            onClick={() => setActiveSubTab('lstm')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-bold transition-all ${
              activeSubTab === 'lstm'
                ? 'bg-burgundy-800 text-gold-200 border border-gold-500/40 shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Cpu className="w-4 h-4 text-emerald-400" />
            <span>LSTM &amp; RNN Gate Studio</span>
          </button>
        </div>
      </div>

      {/* SUB-TAB 1: CNN */}
      {activeSubTab === 'cnn' && (
        <div className="space-y-6">
          
          {/* Section 1: Interactive 2D Convolution */}
          <div className="p-6 rounded-2xl bg-navy-900/90 border border-gold-500/20 space-y-6">
            <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 border-b border-slate-800 pb-4">
              <div>
                <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
                  <Layers className="w-4 h-4 text-gold-400" />
                  <span>Interactive 2D Convolution Operation</span>
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Select a kernel and hover or click any cell in the output feature map to trace the receptive field and arithmetic sum.
                </p>
              </div>

              {/* Kernel Selector */}
              <div className="flex items-center gap-1.5 flex-wrap">
                <span className="text-xs font-bold text-gold-400 mr-1">Kernel:</span>
                {[
                  { id: 'edge', label: 'Laplacian Edge' },
                  { id: 'sobelH', label: 'Sobel Horizontal' },
                  { id: 'sobelV', label: 'Sobel Vertical' },
                  { id: 'sharpen', label: 'Sharpen' },
                  { id: 'blur', label: 'Gaussian Blur' },
                ].map((k) => (
                  <button
                    key={k.id}
                    onClick={() => setKernelType(k.id as any)}
                    className={`px-2.5 py-1 rounded-lg text-xs font-semibold transition-all ${
                      kernelType === k.id
                        ? 'bg-gold-500 text-navy-950 shadow'
                        : 'bg-navy-950 text-slate-300 hover:bg-slate-800'
                    }`}
                  >
                    {k.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Matrix Visualizer Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
              
              {/* Input 5x5 Matrix (col-span-4) */}
              <div className="lg:col-span-4 space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-bold text-gold-300">Input Image X (5×5)</span>
                  <span className="text-[11px] text-slate-400">Receptive Field Highlighted</span>
                </div>
                <div className="grid grid-cols-5 gap-1.5 p-3 rounded-xl bg-navy-950 border border-slate-800 max-w-xs mx-auto">
                  {inputMatrix.map((row, rIdx) =>
                    row.map((val, cIdx) => {
                      const isHighlighted =
                        rIdx >= selectedCell.r &&
                        rIdx < selectedCell.r + 3 &&
                        cIdx >= selectedCell.c &&
                        cIdx < selectedCell.c + 3;

                      return (
                        <div
                          key={`${rIdx}-${cIdx}`}
                          className={`w-11 h-11 rounded-lg flex flex-col items-center justify-center font-mono text-xs font-bold transition-all ${
                            isHighlighted
                              ? 'bg-burgundy-800 text-gold-200 border-2 border-gold-400 shadow-md scale-105'
                              : 'bg-navy-900 text-slate-300 border border-slate-800'
                          }`}
                        >
                          <span>{val}</span>
                          <span className="text-[8px] opacity-50">{rIdx},{cIdx}</span>
                        </div>
                      );
                    })
                  )}
                </div>
              </div>

              {/* Convolution Operator Symbol (col-span-1) */}
              <div className="lg:col-span-1 text-center font-bold text-gold-400 text-2xl flex items-center justify-center">
                *
              </div>

              {/* Kernel 3x3 Matrix (col-span-3) */}
              <div className="lg:col-span-3 space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-bold text-amber-300">Kernel K (3×3)</span>
                  <span className="text-[11px] text-slate-400">{kernelType}</span>
                </div>
                <div className="grid grid-cols-3 gap-1.5 p-3 rounded-xl bg-navy-950 border border-slate-800 max-w-[190px] mx-auto">
                  {currentKernel.map((row, rIdx) =>
                    row.map((val, cIdx) => (
                      <div
                        key={`${rIdx}-${cIdx}`}
                        className="w-11 h-11 rounded-lg flex flex-col items-center justify-center font-mono text-xs font-bold bg-amber-950/40 text-amber-200 border border-amber-500/40"
                      >
                        <span>{val}</span>
                        <span className="text-[8px] opacity-60">{rIdx},{cIdx}</span>
                      </div>
                    ))
                  )}
                </div>
              </div>

              {/* Equals Symbol (col-span-1) */}
              <div className="lg:col-span-1 text-center font-bold text-gold-400 text-2xl flex items-center justify-center">
                =
              </div>

              {/* Output Feature Map 3x3 (col-span-3) */}
              <div className="lg:col-span-3 space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-bold text-emerald-400">Feature Map S (3×3)</span>
                  <span className="text-[11px] text-slate-400">Click to Inspect</span>
                </div>
                <div className="grid grid-cols-3 gap-1.5 p-3 rounded-xl bg-navy-950 border border-slate-800 max-w-[190px] mx-auto">
                  {featureMap.map((row, rIdx) =>
                    row.map((val, cIdx) => {
                      const isSelected = selectedCell.r === rIdx && selectedCell.c === cIdx;
                      return (
                        <div
                          key={`${rIdx}-${cIdx}`}
                          onClick={() => setSelectedCell({ r: rIdx, c: cIdx })}
                          className={`w-11 h-11 rounded-lg flex flex-col items-center justify-center font-mono text-xs font-bold cursor-pointer transition-all ${
                            isSelected
                              ? 'bg-emerald-600 text-slate-950 border-2 border-emerald-300 shadow-lg scale-105'
                              : 'bg-emerald-950/50 text-emerald-300 border border-emerald-500/30 hover:bg-emerald-900/60'
                          }`}
                        >
                          <span>{val}</span>
                          <span className="text-[8px] opacity-60">S[{rIdx},{cIdx}]</span>
                        </div>
                      );
                    })
                  )}
                </div>
              </div>

            </div>

            {/* Step-by-Step Arithmetic Calculation Banner */}
            <div className="p-4 rounded-xl bg-navy-950 border border-gold-500/30 space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="font-bold text-gold-300 uppercase tracking-wide">
                  Arithmetic Derivation for Feature Map Cell S[{patchDetails.r}, {patchDetails.c}]
                </span>
                <span className="font-mono text-emerald-400 font-bold text-sm">
                  Evaluated Value = {patchDetails.total}
                </span>
              </div>
              <p className="text-xs font-mono text-slate-300 bg-navy-900 p-2.5 rounded-lg border border-slate-800 leading-relaxed overflow-x-auto">
                S[{patchDetails.r}, {patchDetails.c}] = {patchDetails.terms.join(' + ')} = <strong className="text-emerald-400">{patchDetails.total}</strong>
              </p>
            </div>
          </div>

          {/* Section 2: Max Pooling vs Average Pooling */}
          <div className="p-6 rounded-2xl bg-navy-900/90 border border-gold-500/20 space-y-4">
            <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
              <Eye className="w-4 h-4 text-emerald-400" />
              <span>Pooling Comparison: Max Pooling vs Average Pooling (4×4 Input, 2×2 Window, Stride 2)</span>
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
              
              {/* 4x4 Input */}
              <div className="space-y-2">
                <div className="text-xs font-bold text-slate-300">Feature Map Input (4×4)</div>
                <div className="grid grid-cols-4 gap-1 p-2 bg-navy-950 rounded-xl border border-slate-800 max-w-[200px] mx-auto">
                  {poolMatrix.map((row, rIdx) =>
                    row.map((val, cIdx) => (
                      <div key={`${rIdx}-${cIdx}`} className="w-10 h-10 rounded bg-navy-900 border border-slate-800 flex items-center justify-center font-mono text-xs font-bold text-slate-200">
                        {val}
                      </div>
                    ))
                  )}
                </div>
              </div>

              {/* Max Pooling Output */}
              <div className="space-y-2">
                <div className="text-xs font-bold text-emerald-400">Max Pooling (2×2, Stride 2)</div>
                <div className="grid grid-cols-2 gap-1.5 p-3 bg-navy-950 rounded-xl border border-emerald-500/30 max-w-[140px] mx-auto">
                  {maxPool2x2.map((row, rIdx) =>
                    row.map((val, cIdx) => (
                      <div key={`${rIdx}-${cIdx}`} className="w-12 h-12 rounded-lg bg-emerald-950 text-emerald-300 border border-emerald-500 flex items-center justify-center font-mono text-sm font-bold shadow">
                        {val}
                      </div>
                    ))
                  )}
                </div>
                <p className="text-[11px] text-slate-400 text-center">Preserves strongest edge activations</p>
              </div>

              {/* Average Pooling Output */}
              <div className="space-y-2">
                <div className="text-xs font-bold text-amber-400">Average Pooling (2×2, Stride 2)</div>
                <div className="grid grid-cols-2 gap-1.5 p-3 bg-navy-950 rounded-xl border border-amber-500/30 max-w-[140px] mx-auto">
                  {avgPool2x2.map((row, rIdx) =>
                    row.map((val, cIdx) => (
                      <div key={`${rIdx}-${cIdx}`} className="w-12 h-12 rounded-lg bg-amber-950 text-amber-200 border border-amber-500 flex items-center justify-center font-mono text-sm font-bold shadow">
                        {val}
                      </div>
                    ))
                  )}
                </div>
                <p className="text-[11px] text-slate-400 text-center">Smooths features, reduces spatial variance</p>
              </div>

            </div>
          </div>

          {/* Section 3: Output Dimension Formula Calculator */}
          <div className="p-6 rounded-2xl bg-navy-900/90 border border-gold-500/20 space-y-4">
            <h3 className="text-sm font-bold text-gold-300 flex items-center gap-2">
              <Sliders className="w-4 h-4 text-gold-400" />
              <span>KTU Formula Calculator: Convolutional Layer Output Dimensions</span>
            </h3>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-4 rounded-xl bg-navy-950 border border-slate-800">
              <div>
                <label className="text-[11px] font-bold text-slate-400 block mb-1">Input Width (W): {calcWin}</label>
                <input 
                  type="range" min="16" max="512" step="16" value={calcWin}
                  onChange={(e) => setCalcWin(Number(e.target.value))}
                  className="w-full accent-gold-500"
                />
              </div>

              <div>
                <label className="text-[11px] font-bold text-slate-400 block mb-1">Kernel Size (K): {calcK}</label>
                <select 
                  value={calcK} 
                  onChange={(e) => setCalcK(Number(e.target.value))}
                  className="w-full bg-navy-900 border border-slate-700 rounded p-1 text-xs text-slate-200"
                >
                  <option value={1}>1×1</option>
                  <option value={3}>3×3</option>
                  <option value={5}>5×5</option>
                  <option value={7}>7×7</option>
                  <option value={11}>11×11</option>
                </select>
              </div>

              <div>
                <label className="text-[11px] font-bold text-slate-400 block mb-1">Padding (P): {calcP}</label>
                <input 
                  type="range" min="0" max="5" step="1" value={calcP}
                  onChange={(e) => setCalcP(Number(e.target.value))}
                  className="w-full accent-gold-500"
                />
              </div>

              <div>
                <label className="text-[11px] font-bold text-slate-400 block mb-1">Stride (S): {calcS}</label>
                <select 
                  value={calcS} 
                  onChange={(e) => setCalcS(Number(e.target.value))}
                  className="w-full bg-navy-900 border border-slate-700 rounded p-1 text-xs text-slate-200"
                >
                  <option value={1}>1</option>
                  <option value={2}>2</option>
                  <option value={4}>4</option>
                </select>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-burgundy-950/30 border border-gold-500/30 flex flex-col md:flex-row items-center justify-between gap-4">
              <div className="font-mono text-xs text-slate-300">
                Formula: <strong className="text-gold-300">O = ⌊(W - K + 2P) / S⌋ + 1</strong> &rarr; ⌊({calcWin} - {calcK} + 2×{calcP}) / {calcS}⌋ + 1
              </div>
              <div className="text-right">
                <span className="text-xs text-slate-400 block">Computed Feature Map Spatial Dim</span>
                <span className="text-2xl font-bold font-serif text-gold-200">{calculatedOutputDim} × {calculatedOutputDim}</span>
              </div>
            </div>
          </div>

        </div>
      )}

      {/* SUB-TAB 2: LSTM */}
      {activeSubTab === 'lstm' && (
        <div className="space-y-6">
          
          <div className="p-6 rounded-2xl bg-navy-900/90 border border-gold-500/20 space-y-6">
            <div className="border-b border-slate-800 pb-4">
              <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
                <Cpu className="w-4 h-4 text-emerald-400" />
                <span>Interactive Long Short-Term Memory (LSTM) Cell Simulator</span>
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Adjust the gate activation levels to observe how the additive cell carousel preserves gradient information across arbitrary time horizons.
              </p>
            </div>

            {/* Sliders Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              
              {/* Previous State Inputs */}
              <div className="p-4 rounded-xl bg-navy-950 border border-slate-800 space-y-4">
                <h4 className="text-xs font-bold uppercase tracking-wider text-gold-300">Previous Time Step (t-1)</h4>
                
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-400">Previous Cell State (C_t-1):</span>
                    <span className="font-mono text-gold-300">{cPrev.toFixed(2)}</span>
                  </div>
                  <input
                    type="range" min="-2" max="2" step="0.05" value={cPrev}
                    onChange={(e) => setCPrev(Number(e.target.value))}
                    className="w-full accent-gold-500"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-400">Previous Hidden State (h_t-1):</span>
                    <span className="font-mono text-gold-300">{hPrev.toFixed(2)}</span>
                  </div>
                  <input
                    type="range" min="-1" max="1" step="0.05" value={hPrev}
                    onChange={(e) => setHPrev(Number(e.target.value))}
                    className="w-full accent-gold-500"
                  />
                </div>
              </div>

              {/* Gate Controls */}
              <div className="p-4 rounded-xl bg-navy-950 border border-slate-800 space-y-3 md:col-span-2">
                <h4 className="text-xs font-bold uppercase tracking-wider text-emerald-400">Current Gating Operations (t)</h4>
                
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-slate-300 font-semibold">Forget Gate f_t:</span>
                      <span className="font-mono text-amber-400 font-bold">{forgetGate.toFixed(2)}</span>
                    </div>
                    <input
                      type="range" min="0" max="1" step="0.01" value={forgetGate}
                      onChange={(e) => setForgetGate(Number(e.target.value))}
                      className="w-full accent-amber-500"
                    />
                    <span className="text-[10px] text-slate-500 block mt-0.5">0 = Erase past, 1 = Full memory</span>
                  </div>

                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-slate-300 font-semibold">Input Gate i_t:</span>
                      <span className="font-mono text-emerald-400 font-bold">{inputGate.toFixed(2)}</span>
                    </div>
                    <input
                      type="range" min="0" max="1" step="0.01" value={inputGate}
                      onChange={(e) => setInputGate(Number(e.target.value))}
                      className="w-full accent-emerald-500"
                    />
                    <span className="text-[10px] text-slate-500 block mt-0.5">Controls write rate of new candidate</span>
                  </div>

                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-slate-300 font-semibold">Candidate State C~_t:</span>
                      <span className="font-mono text-sky-400 font-bold">{candC.toFixed(2)}</span>
                    </div>
                    <input
                      type="range" min="-1" max="1" step="0.05" value={candC}
                      onChange={(e) => setCandC(Number(e.target.value))}
                      className="w-full accent-sky-500"
                    />
                    <span className="text-[10px] text-slate-500 block mt-0.5">tanh activation output [-1, +1]</span>
                  </div>

                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-slate-300 font-semibold">Output Gate o_t:</span>
                      <span className="font-mono text-purple-400 font-bold">{outputGate.toFixed(2)}</span>
                    </div>
                    <input
                      type="range" min="0" max="1" step="0.01" value={outputGate}
                      onChange={(e) => setOutputGate(Number(e.target.value))}
                      className="w-full accent-purple-500"
                    />
                    <span className="text-[10px] text-slate-500 block mt-0.5">Emits filtered cell state to h_t</span>
                  </div>
                </div>
              </div>

            </div>

            {/* Live Evaluated State Display */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              
              <div className="p-5 rounded-xl bg-gradient-to-br from-navy-950 to-burgundy-950/60 border border-gold-500/40 space-y-2">
                <span className="text-xs font-bold uppercase tracking-wider text-gold-300">
                  New Updated Cell State C_t
                </span>
                <div className="text-3xl font-bold font-mono text-gold-200">
                  {cCur.toFixed(4)}
                </div>
                <div className="text-xs font-mono text-slate-400">
                  C_t = ({forgetGate.toFixed(2)} × {cPrev.toFixed(2)}) + ({inputGate.toFixed(2)} × {candC.toFixed(2)}) = {(forgetGate * cPrev).toFixed(3)} + {(inputGate * candC).toFixed(3)}
                </div>
                <p className="text-[11px] text-slate-400 pt-1">
                  Notice the <strong>linear addition</strong>! Gradients can backpropagate through the conveyor belt without exponential attenuation.
                </p>
              </div>

              <div className="p-5 rounded-xl bg-gradient-to-br from-navy-950 to-emerald-950/50 border border-emerald-500/40 space-y-2">
                <span className="text-xs font-bold uppercase tracking-wider text-emerald-300">
                  Emitted Hidden State h_t
                </span>
                <div className="text-3xl font-bold font-mono text-emerald-200">
                  {hCur.toFixed(4)}
                </div>
                <div className="text-xs font-mono text-slate-400">
                  h_t = {outputGate.toFixed(2)} × tanh({cCur.toFixed(3)}) = {outputGate.toFixed(2)} × {Math.tanh(cCur).toFixed(3)}
                </div>
                <p className="text-[11px] text-slate-400 pt-1">
                  The hidden state serves as both the recurrent connection to step (t+1) and the prediction layer output.
                </p>
              </div>

            </div>

          </div>

        </div>
      )}

    </div>
  );
};

