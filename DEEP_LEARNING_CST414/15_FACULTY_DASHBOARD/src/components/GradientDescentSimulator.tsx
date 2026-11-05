import React, { useRef, useEffect, useState } from 'react';
import { Play, RotateCcw, Activity, Zap, CheckCircle2 } from 'lucide-react';

type OptimizerType = 'gd' | 'momentum' | 'nag' | 'adagrad' | 'rmsprop' | 'adam';

interface TrajectoryPoint {
  x: number;
  y: number;
  loss: number;
}

export const GradientDescentSimulator: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [selectedOpt, setSelectedOpt] = useState<OptimizerType>('adam');
  const [lr, setLr] = useState<number>(0.04);
  const [beta, setBeta] = useState<number>(0.9);
  const [isSimulating, setIsSimulating] = useState<boolean>(false);
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [currentLoss, setCurrentLoss] = useState<number>(0);
  const [currentCoords, setCurrentCoords] = useState<{ x: number; y: number }>({ x: -3.5, y: 2.5 });

  // Elliptical ravine loss: f(x, y) = 0.5 * (x^2 + 25 * y^2)
  const lossFunc = (x: number, y: number) => 0.5 * (x * x + 25.0 * y * y);
  const gradFunc = (x: number, y: number) => ({ dx: x, dy: 25.0 * y });

  const trajectoryRef = useRef<TrajectoryPoint[]>([]);
  const stateRef = useRef<{
    pos: { x: number; y: number };
    vel: { vx: number; vy: number };
    sqGrad: { sx: number; sy: number };
    m: { mx: number; my: number };
    v: { vx: number; vy: number };
    step: number;
  }>({
    pos: { x: -3.5, y: 2.5 },
    vel: { vx: 0, vy: 0 },
    sqGrad: { sx: 0, sy: 0 },
    m: { mx: 0, my: 0 },
    v: { vx: 0, vy: 0 },
    step: 0
  });

  const resetSimulation = () => {
    stateRef.current = {
      pos: { x: -3.5, y: 2.5 },
      vel: { vx: 0, vy: 0 },
      sqGrad: { sx: 0, sy: 0 },
      m: { mx: 0, my: 0 },
      v: { vx: 0, vy: 0 },
      step: 0
    };
    trajectoryRef.current = [{ x: -3.5, y: 2.5, loss: lossFunc(-3.5, 2.5) }];
    setCurrentStep(0);
    setCurrentLoss(lossFunc(-3.5, 2.5));
    setCurrentCoords({ x: -3.5, y: 2.5 });
    setIsSimulating(false);
    drawScene();
  };

  const drawScene = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // Coordinate mapping: x in [-5, 5], y in [-3.5, 3.5]
    const toCanvasX = (x: number) => ((x + 5) / 10) * width;
    const toCanvasY = (y: number) => height - ((y + 3.5) / 7) * height;

    // Draw Contour Ellipses
    const levels = [0.2, 0.8, 2.5, 6.0, 12.0, 25.0, 45.0, 75.0];
    levels.forEach((val) => {
      ctx.beginPath();
      ctx.strokeStyle = 'rgba(212, 175, 55, 0.2)';
      ctx.lineWidth = 1;

      // x^2 + 25*y^2 = 2*val => a = sqrt(2*val), b = sqrt(2*val / 25)
      const a = Math.sqrt(2 * val);
      const b = Math.sqrt((2 * val) / 25.0);

      const cx = toCanvasX(0);
      const cy = toCanvasY(0);
      const rx = (a / 10) * width;
      const ry = (b / 7) * height;

      ctx.ellipse(cx, cy, rx, ry, 0, 0, 2 * Math.PI);
      ctx.stroke();
    });

    // Draw Axes
    ctx.strokeStyle = 'rgba(71, 85, 105, 0.4)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, toCanvasY(0));
    ctx.lineTo(width, toCanvasY(0));
    ctx.moveTo(toCanvasX(0), 0);
    ctx.lineTo(toCanvasX(0), height);
    ctx.stroke();

    // Draw Trajectory Path
    if (trajectoryRef.current.length > 1) {
      ctx.beginPath();
      const optColors: Record<OptimizerType, string> = {
        gd: '#EF4444',
        momentum: '#F59E0B',
        nag: '#3B82F6',
        adagrad: '#A855F7',
        rmsprop: '#EC4899',
        adam: '#D4AF37'
      };
      ctx.strokeStyle = optColors[selectedOpt] || '#D4AF37';
      ctx.lineWidth = 2.5;

      const p0 = trajectoryRef.current[0];
      ctx.moveTo(toCanvasX(p0.x), toCanvasY(p0.y));
      for (let i = 1; i < trajectoryRef.current.length; i++) {
        const pt = trajectoryRef.current[i];
        ctx.lineTo(toCanvasX(pt.x), toCanvasY(pt.y));
      }
      ctx.stroke();

      // Trajectory Dots
      trajectoryRef.current.forEach((pt, i) => {
        ctx.beginPath();
        ctx.fillStyle = i === trajectoryRef.current.length - 1 ? '#FDF7E7' : optColors[selectedOpt];
        ctx.arc(toCanvasX(pt.x), toCanvasY(pt.y), i === trajectoryRef.current.length - 1 ? 5 : 2.5, 0, 2 * Math.PI);
        ctx.fill();
      });
    }

    // Draw Target Global Minimum
    ctx.beginPath();
    ctx.fillStyle = '#10B981';
    ctx.arc(toCanvasX(0), toCanvasY(0), 4, 0, 2 * Math.PI);
    ctx.fill();
  };

  useEffect(() => {
    resetSimulation();
  }, [selectedOpt]);

  useEffect(() => {
    let intervalId: any;
    if (isSimulating) {
      intervalId = setInterval(() => {
        const s = stateRef.current;
        if (s.step >= 120 || (Math.abs(s.pos.x) < 0.05 && Math.abs(s.pos.y) < 0.05)) {
          setIsSimulating(false);
          return;
        }

        s.step += 1;
        const g = gradFunc(s.pos.x, s.pos.y);

        if (selectedOpt === 'gd') {
          s.pos.x -= lr * g.dx;
          s.pos.y -= lr * g.dy;
        } else if (selectedOpt === 'momentum') {
          s.vel.vx = beta * s.vel.vx - lr * g.dx;
          s.vel.vy = beta * s.vel.vy - lr * g.dy;
          s.pos.x += s.vel.vx;
          s.pos.y += s.vel.vy;
        } else if (selectedOpt === 'nag') {
          const aheadX = s.pos.x + beta * s.vel.vx;
          const aheadY = s.pos.y + beta * s.vel.vy;
          const gAhead = gradFunc(aheadX, aheadY);
          s.vel.vx = beta * s.vel.vx - lr * gAhead.dx;
          s.vel.vy = beta * s.vel.vy - lr * gAhead.dy;
          s.pos.x += s.vel.vx;
          s.pos.y += s.vel.vy;
        } else if (selectedOpt === 'adagrad') {
          s.sqGrad.sx += g.dx * g.dx;
          s.sqGrad.sy += g.dy * g.dy;
          s.pos.x -= (lr / (Math.sqrt(s.sqGrad.sx) + 1e-8)) * g.dx;
          s.pos.y -= (lr / (Math.sqrt(s.sqGrad.sy) + 1e-8)) * g.dy;
        } else if (selectedOpt === 'rmsprop') {
          s.sqGrad.sx = beta * s.sqGrad.sx + (1 - beta) * (g.dx * g.dx);
          s.sqGrad.sy = beta * s.sqGrad.sy + (1 - beta) * (g.dy * g.dy);
          s.pos.x -= (lr / (Math.sqrt(s.sqGrad.sx) + 1e-8)) * g.dx;
          s.pos.y -= (lr / (Math.sqrt(s.sqGrad.sy) + 1e-8)) * g.dy;
        } else if (selectedOpt === 'adam') {
          const beta1 = 0.9;
          const beta2 = 0.999;
          s.m.mx = beta1 * s.m.mx + (1 - beta1) * g.dx;
          s.m.my = beta1 * s.m.my + (1 - beta1) * g.dy;
          s.v.vx = beta2 * s.v.vx + (1 - beta2) * (g.dx * g.dx);
          s.v.vy = beta2 * s.v.vy + (1 - beta2) * (g.dy * g.dy);
          const mHatX = s.m.mx / (1 - Math.pow(beta1, s.step));
          const mHatY = s.m.my / (1 - Math.pow(beta1, s.step));
          const vHatX = s.v.vx / (1 - Math.pow(beta2, s.step));
          const vHatY = s.v.vy / (1 - Math.pow(beta2, s.step));
          s.pos.x -= (lr / (Math.sqrt(vHatX) + 1e-8)) * mHatX;
          s.pos.y -= (lr / (Math.sqrt(vHatY) + 1e-8)) * mHatY;
        }

        const currentL = lossFunc(s.pos.x, s.pos.y);
        trajectoryRef.current.push({ x: s.pos.x, y: s.pos.y, loss: currentL });
        setCurrentStep(s.step);
        setCurrentLoss(currentL);
        setCurrentCoords({ x: s.pos.x, y: s.pos.y });
        drawScene();
      }, 50);
    }
    return () => clearInterval(intervalId);
  }, [isSimulating, selectedOpt, lr, beta]);

  return (
    <div className="glass-panel rounded-2xl p-6 border border-gold-500/30">
      <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between pb-4 border-b border-slate-800 gap-4">
        <div>
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-gold-400" />
            <h3 className="text-base font-serif font-bold text-gold-300">
              Interactive Optimization Lab: Gradient Descent on Ravines
            </h3>
          </div>
          <p className="text-xs text-slate-400">
            Compare GD, Momentum, Nesterov, AdaGrad, RMSProp, and Adam on an anisotropic ill-conditioned quadratic canyon
          </p>
        </div>

        {/* Live Metrics */}
        <div className="flex items-center gap-3 bg-navy-950 px-3 py-1.5 rounded-xl border border-slate-800 text-xs font-mono">
          <div>Step: <span className="text-gold-300 font-bold">{currentStep}</span></div>
          <div className="text-slate-600">|</div>
          <div>Loss: <span className="text-amber-400 font-bold">{currentLoss.toFixed(4)}</span></div>
          <div className="text-slate-600">|</div>
          <div>Pos: <span className="text-sky-300">({currentCoords.x.toFixed(2)}, {currentCoords.y.toFixed(2)})</span></div>
        </div>
      </div>

      {/* Simulator Controls & Canvas */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mt-4">
        {/* Left Control Panel */}
        <div className="space-y-4">
          <div>
            <label className="text-xs text-slate-300 font-semibold block mb-1.5">Select Optimizer:</label>
            <div className="grid grid-cols-2 gap-1.5">
              {[
                { id: 'gd', label: 'Vanilla GD', color: 'border-red-500/50 text-red-300' },
                { id: 'momentum', label: 'Momentum', color: 'border-amber-500/50 text-amber-300' },
                { id: 'nag', label: 'NAG (Nesterov)', color: 'border-sky-500/50 text-sky-300' },
                { id: 'adagrad', label: 'AdaGrad', color: 'border-purple-500/50 text-purple-300' },
                { id: 'rmsprop', label: 'RMSProp', color: 'border-pink-500/50 text-pink-300' },
                { id: 'adam', label: 'Adam (Default)', color: 'border-gold-500/50 text-gold-300' },
              ].map((opt) => (
                <button
                  key={opt.id}
                  onClick={() => setSelectedOpt(opt.id as OptimizerType)}
                  className={`px-2.5 py-1.5 rounded-lg text-xs font-semibold text-left border transition-all ${
                    selectedOpt === opt.id
                      ? 'bg-burgundy-900/80 border-gold-500 text-gold-200 shadow-md'
                      : 'bg-navy-900/60 border-slate-800 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          <div>
            <div className="flex justify-between text-xs text-slate-300 mb-1">
              <span>Learning Rate (eta):</span>
              <span className="text-gold-400 font-mono">{lr.toFixed(3)}</span>
            </div>
            <input 
              type="range" min="0.005" max="0.10" step="0.005" value={lr}
              onChange={(e) => setLr(parseFloat(e.target.value))}
              className="w-full accent-gold-500"
            />
          </div>

          <div>
            <div className="flex justify-between text-xs text-slate-300 mb-1">
              <span>Momentum / Beta Factor:</span>
              <span className="text-gold-400 font-mono">{beta.toFixed(2)}</span>
            </div>
            <input 
              type="range" min="0.5" max="0.99" step="0.01" value={beta}
              onChange={(e) => setBeta(parseFloat(e.target.value))}
              className="w-full accent-gold-500"
            />
          </div>

          <div className="flex items-center gap-2 pt-2">
            <button
              onClick={() => setIsSimulating(!isSimulating)}
              className="flex-1 py-2 bg-gradient-to-r from-burgundy-700 to-burgundy-900 hover:from-burgundy-600 hover:to-burgundy-800 border border-gold-500/40 rounded-xl text-xs font-bold text-gold-200 shadow-lg flex items-center justify-center gap-1.5"
            >
              {isSimulating ? <Zap className="w-4 h-4 text-amber-400" /> : <Play className="w-4 h-4 text-gold-400" />}
              <span>{isSimulating ? 'Pause Trace' : 'Run Descent'}</span>
            </button>
            <button
              onClick={resetSimulation}
              className="p-2 bg-navy-900 hover:bg-navy-800 border border-slate-700 rounded-xl text-slate-300"
              title="Reset Position"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>

          <div className="p-3 bg-navy-950/80 rounded-xl border border-slate-800 text-[11px] text-slate-300 space-y-1">
            <div className="font-semibold text-gold-400">Classroom Observation:</div>
            <p>
              • <strong>Vanilla GD:</strong> Oscillates violently between steep canyon walls (y-axis) while making slow progress along x-axis.
            </p>
            <p>
              • <strong>Momentum & Adam:</strong> Cancel orthogonal oscillations and accelerate along the valley floor to reach the green minimum.
            </p>
          </div>
        </div>

        {/* Right Canvas */}
        <div className="lg:col-span-3 bg-navy-950 rounded-xl border border-slate-800 p-2 flex items-center justify-center relative">
          <canvas 
            ref={canvasRef} 
            width={720} 
            height={380} 
            className="w-full h-auto rounded-lg"
          />
        </div>
      </div>
    </div>
  );
};

