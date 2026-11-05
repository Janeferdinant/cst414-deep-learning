import React, { useState } from 'react';
import { 
  GraduationCap, Award, FileCheck, CheckCircle2, 
  BarChart3, Target, Calendar, ChevronRight, Download
} from 'lucide-react';

export const AssessmentsView: React.FC = () => {
  const [activeSubTab, setActiveSubTab] = useState<'cie' | 'cos' | 'assignments' | 'capstone'>('cie');

  // Course Outcomes
  const courseOutcomes = [
    {
      code: 'CO1',
      title: 'Neural Network Foundations',
      description: 'Understand the mathematical foundations of biological and artificial neural networks, multilayer perceptrons, backpropagation, and convergence pathologies.',
      blooms: 'K2 (Understand)',
      hours: 7,
      cieMarks: 10,
      eseMarks: 17,
      pos: { PO1: 3, PO2: 3, PO3: 2, PO4: 1, PO5: 2, PSO1: 3, PSO2: 2 }
    },
    {
      code: 'CO2',
      title: 'Deep Learning Optimization & Regularization',
      description: 'Apply advanced optimization techniques (Momentum, NAG, Adam) and regularization strategies (L1/L2, Dropout, Early Stopping) to train deep neural architectures.',
      blooms: 'K3 (Apply)',
      hours: 9,
      cieMarks: 15,
      eseMarks: 20,
      pos: { PO1: 3, PO2: 3, PO3: 3, PO4: 2, PO5: 3, PSO1: 3, PSO2: 3 }
    },
    {
      code: 'CO3',
      title: 'Convolutional Neural Networks',
      description: 'Design and analyze convolutional layers, pooling mechanisms, receptive fields, and efficient algorithms (Winograd, MobileNet) for spatial computer vision tasks.',
      blooms: 'K3 (Apply)',
      hours: 6,
      cieMarks: 10,
      eseMarks: 20,
      pos: { PO1: 3, PO2: 3, PO3: 3, PO4: 2, PO5: 3, PSO1: 3, PSO2: 3 }
    },
    {
      code: 'CO4',
      title: 'Recurrent Neural Networks & Memory Units',
      description: 'Formulate sequence processing models using unrolled computational graphs, BPTT, Encoder-Decoder Seq2Seq, and gated units (LSTM, GRU) for temporal series.',
      blooms: 'K3 (Apply)',
      hours: 5,
      cieMarks: 8,
      eseMarks: 20,
      pos: { PO1: 3, PO2: 3, PO3: 3, PO4: 2, PO5: 3, PSO1: 3, PSO2: 3 }
    },
    {
      code: 'CO5',
      title: 'Representation Learning & Generative Models',
      description: 'Implement distributed word representations (Word2Vec, GloVe), undercomplete autoencoders, and energy-based probabilistic models (RBM, DBN).',
      blooms: 'K3 (Apply)',
      hours: 9,
      cieMarks: 7,
      eseMarks: 23,
      pos: { PO1: 3, PO2: 3, PO3: 3, PO4: 3, PO5: 3, PSO1: 3, PSO2: 3 }
    }
  ];

  // 5 Formal Course Assignments
  const assignments = [
    {
      mod: 1,
      title: 'Assignment 1: Perceptron Convergence & Backpropagation Derivation',
      deadline: '10-Sep-2025 (Provisional)',
      co: 'CO1',
      marks: 15,
      questions: [
        '1. Implement the Perceptron Learning Algorithm from scratch in Python and mathematically prove convergence for linearly separable data.',
        '2. Hand-derive the full backpropagation gradients for a 3-layer MLP using Cross-Entropy loss with Softmax output.',
        '3. Compare the vanishing gradient behaviors of Sigmoid, Tanh, and Leaky ReLU activations across 10 hidden layers.'
      ],
      rubric: 'Mathematical rigor: 40% | Code implementation: 35% | Analysis & Discussion: 25%'
    },
    {
      mod: 2,
      title: 'Assignment 2: Deep Optimization Benchmark & Regularization Ablation',
      deadline: '01-Oct-2025 (Provisional)',
      co: 'CO2',
      marks: 15,
      questions: [
        '1. Implement and compare 6 optimizers (SGD, Momentum, NAG, AdaGrad, RMSProp, Adam) on Beale and Rosenbrock loss surfaces.',
        '2. Conduct an ablation study on CIFAR-10 comparing L2 regularization (weight decay), Dropout (p=0.2, 0.5), and Early Stopping.',
        '3. Derive the mathematical bias-correction equations for the first and second moment vectors in Adam.'
      ],
      rubric: 'Optimizer benchmarking: 40% | Regularization experiment: 35% | Derivation & formatting: 25%'
    },
    {
      mod: 3,
      title: 'Assignment 3: Custom CNN Architecture & Receptive Field Analysis',
      deadline: '22-Oct-2025 (Provisional)',
      co: 'CO3',
      marks: 15,
      questions: [
        '1. Design a 5-layer CNN for Fashion-MNIST achieving >92% test accuracy; calculate parameter count and FLOPs for every layer.',
        '2. Hand-calculate the effective receptive field of the final feature map with respect to the input image.',
        '3. Implement Winograd convolution for a 1D signal with filter size r=3 and tile m=2; verify exact arithmetic equivalence.'
      ],
      rubric: 'Architecture design: 35% | Parameter/Receptive field arithmetic: 35% | Winograd implementation: 30%'
    },
    {
      mod: 4,
      title: 'Assignment 4: Gated Recurrent Network for Sequence Modeling',
      deadline: '05-Nov-2025 (Provisional)',
      co: 'CO4',
      marks: 15,
      questions: [
        '1. Implement a character-level text generator using vanilla RNN, LSTM, and GRU; contrast training loss curves and perplexity.',
        '2. Diagram the exact computational forward and backward graph for an LSTM cell at time step t.',
        '3. Build a Seq2Seq model with Teacher Forcing for English-to-French character transliteration.'
      ],
      rubric: 'Sequence generation code: 40% | LSTM mathematical graph: 30% | Seq2Seq implementation: 30%'
    },
    {
      mod: 5,
      title: 'Assignment 5: Word Embeddings & Variational Autoencoder',
      deadline: '19-Nov-2025 (Provisional)',
      co: 'CO5',
      marks: 15,
      questions: [
        '1. Train a Skip-Gram Word2Vec model with Negative Sampling on a 1MB text corpus; evaluate semantic vector analogies (king - man + woman = queen).',
        '2. Implement an Undercomplete Autoencoder on MNIST and visualize the 2D latent space manifold.',
        '3. Derive the Contrastive Divergence update rule for Restricted Boltzmann Machines (RBM).'
      ],
      rubric: 'Embedding training & analogies: 40% | Autoencoder latent manifold: 35% | RBM derivation: 25%'
    }
  ];

  return (
    <div className="space-y-6">
      
      {/* Header */}
      <div className="p-6 rounded-2xl bg-gradient-to-r from-navy-900 via-burgundy-950/60 to-navy-900 border border-gold-500/30 flex flex-col md:flex-row items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-burgundy-800 text-gold-300 border border-gold-500/30">
              Outcome-Based Education (OBE)
            </span>
            <span className="text-xs text-slate-400 font-medium">KTU B.Tech Scheme 2019 Assessment System</span>
          </div>
          <h2 className="text-lg md:text-xl font-bold font-serif text-slate-100 mt-1">
            Continuous Internal Evaluation (CIE), Course Outcomes &amp; Attainment Mapping
          </h2>
          <p className="text-xs text-slate-300 mt-1">
            Comprehensive evaluation framework covering 50 CIE marks, 100 ESE marks, 5 COs, and ABET/NBA PO matrix.
          </p>
        </div>

        {/* Tab switch */}
        <div className="flex items-center bg-navy-950 rounded-xl p-1 border border-gold-500/30 shadow-lg">
          <button
            onClick={() => setActiveSubTab('cie')}
            className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
              activeSubTab === 'cie' ? 'bg-burgundy-800 text-gold-200 border border-gold-500/40 shadow' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            CIE Scheme (50M)
          </button>
          <button
            onClick={() => setActiveSubTab('cos')}
            className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
              activeSubTab === 'cos' ? 'bg-burgundy-800 text-gold-200 border border-gold-500/40 shadow' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            CO-PO Matrix
          </button>
          <button
            onClick={() => setActiveSubTab('assignments')}
            className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
              activeSubTab === 'assignments' ? 'bg-burgundy-800 text-gold-200 border border-gold-500/40 shadow' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            5 Assignments
          </button>
          <button
            onClick={() => setActiveSubTab('capstone')}
            className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
              activeSubTab === 'capstone' ? 'bg-burgundy-800 text-gold-200 border border-gold-500/40 shadow' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Capstone Brief
          </button>
        </div>
      </div>

      {/* SUB-TAB 1: CIE SCHEME */}
      {activeSubTab === 'cie' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            
            {/* Card 1: Attendance */}
            <div className="p-5 rounded-xl bg-navy-900/90 border border-gold-500/20 space-y-3">
              <div className="flex items-center justify-between">
                <span className="px-2 py-0.5 rounded text-[11px] font-bold bg-navy-950 text-gold-400 border border-slate-700">
                  CIE Component 1
                </span>
                <span className="text-xl font-bold font-serif text-slate-100">10 Marks</span>
              </div>
              <h3 className="text-sm font-bold text-slate-200">Attendance Norms</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Evaluated as per KTU B.Tech Ordinance Section 12:
              </p>
              <ul className="text-xs text-slate-300 space-y-1">
                <li className="flex justify-between"><span>&ge; 90% attendance:</span> <strong className="text-emerald-400">10 Marks</strong></li>
                <li className="flex justify-between"><span>85% - 89% attendance:</span> <strong className="text-slate-200">9 Marks</strong></li>
                <li className="flex justify-between"><span>80% - 84% attendance:</span> <strong className="text-slate-200">8 Marks</strong></li>
                <li className="flex justify-between"><span>75% - 79% attendance:</span> <strong className="text-amber-400">7 Marks</strong></li>
                <li className="flex justify-between"><span>&lt; 75% attendance:</span> <strong className="text-red-400">Ineligible (FE)</strong></li>
              </ul>
            </div>

            {/* Card 2: Internal Tests */}
            <div className="p-5 rounded-xl bg-navy-900/90 border border-gold-500/20 space-y-3">
              <div className="flex items-center justify-between">
                <span className="px-2 py-0.5 rounded text-[11px] font-bold bg-navy-950 text-gold-400 border border-slate-700">
                  CIE Component 2
                </span>
                <span className="text-xl font-bold font-serif text-slate-100">25 Marks</span>
              </div>
              <h3 className="text-sm font-bold text-slate-200">Continuous Assessment Tests</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Two central Series Tests of 50 marks each (reduced to 25 marks average):
              </p>
              <ul className="text-xs text-slate-300 space-y-1.5">
                <li>
                  <strong className="text-gold-300">Series Test 1:</strong> 50 Marks (Modules 1 &amp; 2), 2 hours duration.
                </li>
                <li>
                  <strong className="text-gold-300">Series Test 2:</strong> 50 Marks (Modules 3, 4 &amp; 5), 2 hours duration.
                </li>
                <li className="text-[11px] text-slate-400 pt-1">
                  Average = (Test 1 + Test 2) / 4 = 25 Marks maximum.
                </li>
              </ul>
            </div>

            {/* Card 3: Assignments */}
            <div className="p-5 rounded-xl bg-navy-900/90 border border-gold-500/20 space-y-3">
              <div className="flex items-center justify-between">
                <span className="px-2 py-0.5 rounded text-[11px] font-bold bg-navy-950 text-gold-400 border border-slate-700">
                  CIE Component 3
                </span>
                <span className="text-xl font-bold font-serif text-slate-100">15 Marks</span>
              </div>
              <h3 className="text-sm font-bold text-slate-200">Assignments &amp; Micro-Projects</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Two graded analytical assignments and one software implementation:
              </p>
              <ul className="text-xs text-slate-300 space-y-1.5">
                <li><strong className="text-gold-300">Assignment 1:</strong> 5 Marks (Written derivation &amp; proof)</li>
                <li><strong className="text-gold-300">Assignment 2:</strong> 5 Marks (Algorithm benchmark report)</li>
                <li><strong className="text-gold-300">Capstone/Mini-project:</strong> 5 Marks (Executable pipeline)</li>
              </ul>
            </div>

          </div>

          {/* ESE Summary Banner */}
          <div className="p-5 rounded-xl bg-burgundy-950/30 border border-gold-500/30 flex flex-col md:flex-row items-center justify-between gap-4">
            <div>
              <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-burgundy-900 text-gold-300 border border-gold-500/30">
                End Semester Examination (ESE)
              </span>
              <h4 className="text-base font-bold text-slate-100 font-serif mt-1">
                KTU CST414 University Examination: 100 Marks (3 Hours)
              </h4>
              <p className="text-xs text-slate-300 mt-0.5">
                Minimum 40% (40/100) in ESE and 45% (45/150) total aggregate required to pass course with grade P or higher.
              </p>
            </div>
            <div className="text-right">
              <span className="text-xs text-slate-400 block">Total Course Weightage</span>
              <span className="text-2xl font-bold font-serif text-gold-300">150 Marks (3 Credits)</span>
            </div>
          </div>
        </div>
      )}

      {/* SUB-TAB 2: CO-PO MATRIX */}
      {activeSubTab === 'cos' && (
        <div className="space-y-6">
          <div className="rounded-xl border border-gold-500/20 bg-navy-900/80 overflow-hidden shadow-lg">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="bg-navy-950 text-gold-300 uppercase tracking-wider font-semibold border-b border-gold-500/20">
                  <tr>
                    <th className="py-3 px-3 w-16 text-center">CO Code</th>
                    <th className="py-3 px-4">Course Outcome Statement</th>
                    <th className="py-3 px-3 w-28 text-center">Bloom's Level</th>
                    <th className="py-3 px-3 w-16 text-center">Hours</th>
                    <th className="py-3 px-2 text-center">PO1</th>
                    <th className="py-3 px-2 text-center">PO2</th>
                    <th className="py-3 px-2 text-center">PO3</th>
                    <th className="py-3 px-2 text-center">PO4</th>
                    <th className="py-3 px-2 text-center">PO5</th>
                    <th className="py-3 px-2 text-center">PSO1</th>
                    <th className="py-3 px-2 text-center">PSO2</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800">
                  {courseOutcomes.map((co) => (
                    <tr key={co.code} className="hover:bg-slate-800/40 transition-colors">
                      <td className="py-3 px-3 text-center font-bold text-gold-400 bg-navy-950/60">
                        {co.code}
                      </td>
                      <td className="py-3 px-4">
                        <div className="font-semibold text-slate-200">{co.title}</div>
                        <div className="text-[11px] text-slate-400 mt-0.5">{co.description}</div>
                      </td>
                      <td className="py-3 px-3 text-center">
                        <span className="px-2 py-0.5 rounded text-[11px] bg-navy-950 text-slate-300 border border-slate-700">
                          {co.blooms}
                        </span>
                      </td>
                      <td className="py-3 px-3 text-center font-mono font-bold text-slate-200">
                        {co.hours}h
                      </td>
                      <td className="py-3 px-2 text-center font-bold text-emerald-400">{co.pos.PO1}</td>
                      <td className="py-3 px-2 text-center font-bold text-emerald-400">{co.pos.PO2}</td>
                      <td className="py-3 px-2 text-center font-bold text-emerald-400">{co.pos.PO3}</td>
                      <td className="py-3 px-2 text-center font-bold text-amber-400">{co.pos.PO4}</td>
                      <td className="py-3 px-2 text-center font-bold text-emerald-400">{co.pos.PO5}</td>
                      <td className="py-3 px-2 text-center font-bold text-gold-400">{co.pos.PSO1}</td>
                      <td className="py-3 px-2 text-center font-bold text-gold-400">{co.pos.PSO2}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="p-3 bg-navy-950 text-[11px] text-slate-400 flex items-center justify-between border-t border-slate-800">
              <span>Correlation Levels: 3 = Substantial (High) &bull; 2 = Moderate (Medium) &bull; 1 = Slight (Low)</span>
              <span>Complies with NBA Criterion 3 (Course Outcomes &amp; Program Outcomes)</span>
            </div>
          </div>
        </div>
      )}

      {/* SUB-TAB 3: ASSIGNMENTS */}
      {activeSubTab === 'assignments' && (
        <div className="space-y-4">
          {assignments.map((asg) => (
            <div key={asg.mod} className="p-5 rounded-xl bg-navy-900/90 border border-gold-500/20 space-y-3">
              <div className="flex items-center justify-between gap-2 border-b border-slate-800 pb-2.5">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-burgundy-900 text-gold-300 border border-gold-500/30">
                      Module {asg.mod}
                    </span>
                    <span className="text-xs text-slate-400 font-medium">
                      Mapped to {asg.co} &bull; {asg.marks} Marks
                    </span>
                  </div>
                  <h3 className="text-sm font-bold text-slate-100 mt-1">{asg.title}</h3>
                </div>
                <div className="text-right">
                  <span className="text-[11px] text-slate-400 block">Submission Deadline</span>
                  <span className="text-xs font-bold text-amber-300">{asg.deadline}</span>
                </div>
              </div>

              <div className="space-y-1.5 pl-2">
                {asg.questions.map((q, qIdx) => (
                  <p key={qIdx} className="text-xs text-slate-300 leading-relaxed font-sans">
                    {q}
                  </p>
                ))}
              </div>

              <div className="p-2.5 rounded-lg bg-navy-950 border border-slate-800 text-[11px] text-slate-400 flex items-center justify-between">
                <span><strong className="text-gold-300">Grading Rubric:</strong> {asg.rubric}</span>
                <span className="text-emerald-400 font-medium">Includes LaTeX / Jupyter verification</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* SUB-TAB 4: CAPSTONE */}
      {activeSubTab === 'capstone' && (
        <div className="p-6 rounded-2xl bg-navy-900/90 border border-gold-500/30 space-y-6">
          <div className="border-b border-slate-800 pb-4">
            <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-burgundy-900 text-gold-300 border border-gold-500/30">
              Department Capstone Project
            </span>
            <h3 className="text-lg font-bold font-serif text-slate-100 mt-1">
              End-to-End Real-Time Multi-Modal Deep Learning System
            </h3>
            <p className="text-xs text-slate-400 mt-1">
              Teams of 3-4 students design, train, evaluate, and deploy a production deep learning model solving a real-world societal or industrial problem.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
            <div className="p-4 rounded-xl bg-navy-950 border border-slate-800 space-y-2">
              <h4 className="font-bold text-gold-300 uppercase tracking-wider">Milestone 1 (Week 4)</h4>
              <p className="text-slate-300">Problem formulation, literature review (Goodfellow / arXiv), dataset pipeline selection, and baseline architecture design.</p>
            </div>

            <div className="p-4 rounded-xl bg-navy-950 border border-slate-800 space-y-2">
              <h4 className="font-bold text-gold-300 uppercase tracking-wider">Milestone 2 (Week 8)</h4>
              <p className="text-slate-300">Model training, optimizer ablation (Adam vs RMSProp), regularization (Dropout/L2), loss curves, and intermediate checkpoint demo.</p>
            </div>

            <div className="p-4 rounded-xl bg-navy-950 border border-slate-800 space-y-2">
              <h4 className="font-bold text-gold-300 uppercase tracking-wider">Milestone 3 (Week 12)</h4>
              <p className="text-slate-300">Final evaluation (Confusion matrix, IoU/mAP, Perplexity), Web/API deployment (Streamlit/FastAPI), viva presentation, and technical report.</p>
            </div>
          </div>

          <div className="p-4 rounded-xl bg-burgundy-950/30 border border-gold-500/30 text-xs flex items-center justify-between">
            <div>
              <span className="font-bold text-gold-300 block">Evaluation Committee:</span>
              <span className="text-slate-300">Jane Rose (Course Coordinator) &amp; Department Review Panel</span>
            </div>
            <span className="font-bold text-emerald-400">Weightage: 15 Marks (Part of CIE)</span>
          </div>
        </div>
      )}

    </div>
  );
};

