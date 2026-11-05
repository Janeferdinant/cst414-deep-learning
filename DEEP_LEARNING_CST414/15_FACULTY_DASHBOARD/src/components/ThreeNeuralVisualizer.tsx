import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { Play, Pause, RotateCcw, ZoomIn, ZoomOut, Layers, Info } from 'lucide-react';

export const ThreeNeuralVisualizer: React.FC = () => {
  const mountRef = useRef<HTMLDivElement>(null);
  const [isPlaying, setIsPlaying] = useState(true);
  const [selectedLayer, setSelectedLayer] = useState<string>('All Layers');
  const [pulseSpeed, setPulseSpeed] = useState<number>(1.0);
  const [hoveredNeuron, setHoveredNeuron] = useState<string | null>(null);

  useEffect(() => {
    if (!mountRef.current) return;

    const width = mountRef.current.clientWidth;
    const height = 450;

    // Scene, Camera, Renderer
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x070B19);

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(0, 0, 32);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mountRef.current.appendChild(renderer.domElement);

    // Layer Specifications: Input (4), Hidden1 (6), Hidden2 (5), Output (2)
    const layerDefs = [
      { name: 'Input Layer', count: 4, x: -12, color: 0x38BDF8, label: 'Inputs x1..x4' },
      { name: 'Hidden Layer 1', count: 7, x: -4, color: 0xD4AF37, label: 'Dense (ReLU)' },
      { name: 'Hidden Layer 2', count: 5, x: 4, color: 0xB81D39, label: 'Dense (ReLU)' },
      { name: 'Output Layer', count: 2, x: 12, color: 0x10B981, label: 'Softmax / Logits' }
    ];

    const neurons: THREE.Mesh[] = [];
    const layerNodes: THREE.Vector3[][] = [];

    // Ambient & Point Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(ambientLight);

    const pointLight = new THREE.PointLight(0xD4AF37, 2.5, 100);
    pointLight.position.set(0, 15, 20);
    scene.add(pointLight);

    const burgundyLight = new THREE.PointLight(0xB81D39, 2.0, 100);
    burgundyLight.position.set(0, -15, 20);
    scene.add(burgundyLight);

    // Create Neurons
    layerDefs.forEach((layer, layerIdx) => {
      const nodesInLayer: THREE.Vector3[] = [];
      const spacing = 3.2;
      const yOffset = ((layer.count - 1) * spacing) / 2;

      for (let i = 0; i < layer.count; i++) {
        const y = i * spacing - yOffset;
        const pos = new THREE.Vector3(layer.x, y, 0);
        nodesInLayer.push(pos);

        // Sphere Geometry
        const geom = new THREE.SphereGeometry(0.75, 24, 24);
        const mat = new THREE.MeshPhongMaterial({
          color: layer.color,
          emissive: layer.color,
          emissiveIntensity: 0.35,
          shininess: 90
        });
        const mesh = new THREE.Mesh(geom, mat);
        mesh.position.copy(pos);
        mesh.userData = {
          name: `${layer.name} Node #${i + 1}`,
          layerName: layer.name,
          details: `${layer.label} | Position: (${pos.x}, ${pos.y.toFixed(1)})`
        };
        scene.add(mesh);
        neurons.push(mesh);
      }
      layerNodes.push(nodesInLayer);
    });

    // Create Synaptic Weights (Lines between consecutive layers)
    const lineMat = new THREE.LineBasicMaterial({
      color: 0x475569,
      transparent: true,
      opacity: 0.35
    });

    for (let l = 0; l < layerNodes.length - 1; l++) {
      const currLayer = layerNodes[l];
      const nextLayer = layerNodes[l + 1];

      currLayer.forEach((p1) => {
        nextLayer.forEach((p2) => {
          const points = [p1, p2];
          const lineGeom = new THREE.BufferGeometry().setFromPoints(points);
          const line = new THREE.Line(lineGeom, lineMat);
          scene.add(line);
        });
      });
    }

    // Glowing signal pulses moving along synapses
    const pulseCount = 35;
    const pulseGeometry = new THREE.SphereGeometry(0.18, 12, 12);
    const pulseMaterial = new THREE.MeshBasicMaterial({ color: 0xFDF7E7 });
    const pulses: { mesh: THREE.Mesh; start: THREE.Vector3; end: THREE.Vector3; progress: number; speed: number }[] = [];

    for (let i = 0; i < pulseCount; i++) {
      const l = Math.floor(Math.random() * (layerNodes.length - 1));
      const startNodes = layerNodes[l];
      const endNodes = layerNodes[l + 1];
      const start = startNodes[Math.floor(Math.random() * startNodes.length)];
      const end = endNodes[Math.floor(Math.random() * endNodes.length)];

      const mesh = new THREE.Mesh(pulseGeometry, pulseMaterial);
      mesh.position.copy(start);
      scene.add(mesh);
      pulses.push({
        mesh,
        start,
        end,
        progress: Math.random(),
        speed: 0.006 + Math.random() * 0.008
      });
    }

    // Mouse Interaction
    let mouseX = 0;
    let mouseY = 0;
    let targetRotationX = 0;
    let targetRotationY = 0;

    const handleMouseMove = (e: MouseEvent) => {
      const rect = mountRef.current?.getBoundingClientRect();
      if (!rect) return;
      mouseX = ((e.clientX - rect.left) / width) * 2 - 1;
      mouseY = -(((e.clientY - rect.top) / height) * 2 - 1);
      targetRotationY = mouseX * 0.45;
      targetRotationX = -mouseY * 0.25;
    };

    window.addEventListener('mousemove', handleMouseMove);

    // Animation Loop
    let animationFrameId: number;
    let clock = new THREE.Clock();

    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      const delta = clock.getDelta();

      // Smooth camera orbit
      scene.rotation.y += (targetRotationY - scene.rotation.y) * 0.05;
      scene.rotation.x += (targetRotationX - scene.rotation.x) * 0.05;

      // Animate Neurons Breathing
      const time = clock.getElapsedTime() * pulseSpeed;
      neurons.forEach((n, idx) => {
        const scale = 1.0 + Math.sin(time * 2 + idx * 0.5) * 0.08;
        n.scale.set(scale, scale, scale);
      });

      // Animate Synaptic Pulses
      pulses.forEach((p) => {
        p.progress += p.speed * pulseSpeed;
        if (p.progress >= 1.0) {
          p.progress = 0;
          const l = Math.floor(Math.random() * (layerNodes.length - 1));
          const startNodes = layerNodes[l];
          const endNodes = layerNodes[l + 1];
          p.start = startNodes[Math.floor(Math.random() * startNodes.length)];
          p.end = endNodes[Math.floor(Math.random() * endNodes.length)];
        }
        p.mesh.position.lerpVectors(p.start, p.end, p.progress);
      });

      renderer.render(scene, camera);
    };

    animate();

    const handleResize = () => {
      if (!mountRef.current) return;
      const newWidth = mountRef.current.clientWidth;
      camera.aspect = newWidth / height;
      camera.updateProjectionMatrix();
      renderer.setSize(newWidth, height);
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationFrameId);
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, [pulseSpeed]);

  return (
    <div className="glass-panel rounded-2xl p-6 border border-gold-500/30 relative overflow-hidden">
      {/* Header & Controls */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between pb-4 border-b border-slate-800 gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-ping" />
            <h3 className="text-base font-serif font-bold text-gold-300">
              Interactive 3D Deep Neural Architecture Studio
            </h3>
          </div>
          <p className="text-xs text-slate-400">
            Real-time WebGL rendering of feedforward deep representations, synaptic signal flows, and layer activations
          </p>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 bg-navy-950/80 px-3 py-1.5 rounded-lg border border-slate-800 text-xs">
            <span className="text-slate-400">Speed:</span>
            <input 
              type="range" 
              min="0.2" 
              max="2.5" 
              step="0.1" 
              value={pulseSpeed} 
              onChange={(e) => setPulseSpeed(parseFloat(e.target.value))}
              className="w-16 accent-gold-500"
            />
            <span className="text-gold-300 font-mono text-[10px]">{pulseSpeed.toFixed(1)}x</span>
          </div>

          <div className="flex items-center gap-1 bg-navy-950/80 p-1 rounded-lg border border-slate-800 text-xs text-slate-300">
            <span className="px-2 py-0.5 rounded text-[10px] bg-sky-500/20 text-sky-300">Input: 4</span>
            <span className="px-2 py-0.5 rounded text-[10px] bg-gold-500/20 text-gold-300">H1: 7</span>
            <span className="px-2 py-0.5 rounded text-[10px] bg-burgundy-500/20 text-burgundy-300">H2: 5</span>
            <span className="px-2 py-0.5 rounded text-[10px] bg-emerald-500/20 text-emerald-300">Out: 2</span>
          </div>
        </div>
      </div>

      {/* 3D WebGL Canvas Container */}
      <div 
        ref={mountRef} 
        className="w-full h-[450px] relative rounded-xl overflow-hidden cursor-grab active:cursor-grabbing"
      />

      {/* Floating Info Overlay */}
      <div className="absolute bottom-10 left-10 glass-panel bg-navy-950/90 border border-gold-500/20 p-3 rounded-xl max-w-sm text-xs pointer-events-none">
        <div className="flex items-center gap-1.5 text-gold-300 font-semibold mb-1">
          <Info className="w-3.5 h-3.5" />
          <span>Interactive Orbit & Synaptic Pulse</span>
        </div>
        <p className="text-[11px] text-slate-300 leading-relaxed">
          Move your cursor across the canvas to orbit the 3D architecture. White photon pulses demonstrate forward gradient excitation and backpropagation sensitivity pathways.
        </p>
      </div>
    </div>
  );
};

