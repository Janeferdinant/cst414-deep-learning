import React, { useState } from 'react';
import { 
  X, BookOpen, Clock, Calculator, Code2, Globe, 
  HelpCircle, CheckSquare, Download, ChevronLeft, 
  ChevronRight, CheckCircle2, Copy, Check, FileText,
  AlertCircle, ExternalLink, Sparkles, Award
} from 'lucide-react';
import { ScheduleItem } from './ScheduleView';

interface DayExplorerModalProps {
  dayItem: ScheduleItem | null;
  onClose: () => void;
  onSelectDay: (dayNum: number) => void;
  totalDays: number;
  isCompleted: boolean;
  onToggleStatus: (dayNum: number) => void;
}

export const DayExplorerModal: React.FC<DayExplorerModalProps> = ({
  dayItem,
  onClose,
  onSelectDay,
  totalDays,
  isCompleted,
  onToggleStatus
}) => {
  if (!dayItem) return null;

  const [activeTab, setActiveTab] = useState<
    'notes' | 'script' | 'numericals' | 'demo' | 'application' | 'viva' | 'quiz' | 'files'
  >('notes');
  const [copiedCode, setCopiedCode] = useState(false);
  const [selectedQuizAnswers, setSelectedQuizAnswers] = useState<Record<number, string>>({});

  const dayData = dayItem.data || {};
  const notes = dayData.notes || {};
  const script = dayData.teaching_script || {};
  const demo = dayData.python_demo || {};
  const numericals = dayData.numerical_problems || [];
  const app = dayData.real_world_application || {};
  const viva = dayData.viva_questions || [];
  const assessment = dayData.assessment || {};

  const handleCopyCode = () => {
    if (demo.code) {
      navigator.clipboard.writeText(demo.code);
      setCopiedCode(true);
      setTimeout(() => setCopiedCode(false), 2000);
    }
  };

  const handleQuizSelect = (qIdx: number, option: string) => {
    setSelectedQuizAnswers(prev => ({ ...prev, [qIdx]: option }));
  };

  const tabs = [
    { id: 'notes', label: 'KTU Notes', icon: BookOpen },
    { id: 'script', label: 'Classroom Script', icon: Clock },
    { id: 'numericals', label: 'Solved Numericals', icon: Calculator, badge: numericals.length },
    { id: 'demo', label: 'Python Lab', icon: Code2 },
    { id: 'application', label: 'Industry Case', icon: Globe },
    { id: 'viva', label: 'Viva Voce', icon: HelpCircle, badge: viva.length },
    { id: 'quiz', label: 'Quiz & Rubric', icon: CheckSquare },
    { id: 'files', label: 'Assets & Downloads', icon: Download },
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-3 md:p-6 bg-navy-950/80 backdrop-blur-md overflow-y-auto">
      <div className="bg-navy-900 border border-gold-500/30 rounded-2xl w-full max-w-5xl max-h-[92vh] flex flex-col shadow-2xl shadow-navy-950/90 overflow-hidden my-auto animate-in fade-in duration-200">
        
        {/* Modal Header */}
        <div className="px-6 py-4 bg-gradient-to-r from-navy-950 via-burgundy-950/70 to-navy-950 border-b border-gold-500/20 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <span className="w-10 h-10 rounded-xl bg-burgundy-900/90 border border-gold-500/40 text-gold-300 font-bold font-serif flex items-center justify-center text-sm shadow">
              #{dayItem.day_number.toString().padStart(2, '0')}
            </span>
            <div>
              <div className="flex items-center gap-2 flex-wrap">
                <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-burgundy-800 text-gold-200 border border-gold-500/30">
                  M{dayItem.module} : Topic {dayItem.topic_id}
                </span>
                <span className="text-xs text-slate-400 font-medium">
                  {dayItem.date_display} &bull; {dayItem.day_name} &bull; {dayItem.period}
                </span>
                <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-navy-950 text-gold-300 border border-slate-700">
                  {dayItem.co}
                </span>
                <span className="px-1.5 py-0.5 rounded text-[10px] text-slate-300 bg-slate-900 border border-slate-700">
                  {dayItem.blooms}
                </span>
              </div>
              <h2 className="text-base md:text-lg font-bold text-slate-100 font-serif mt-1">
                {dayItem.title}
              </h2>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* Direct Handwritten PDF Download */}
            <a
              href={`/downloads/CST414_Module_${dayItem.module}_Handwritten_Notes.pdf`}
              download={`CST414_Module_${dayItem.module}_Handwritten_Notes.pdf`}
              className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-burgundy-900 hover:bg-burgundy-800 border border-gold-500/40 text-gold-200 flex items-center gap-1.5 shadow transition-all"
              title="Download Printable Handwritten Notes PDF"
            >
              <Download className="w-3.5 h-3.5 text-gold-400" />
              <span className="hidden sm:inline">Handwritten PDF</span>
            </a>

            {/* Print button */}
            <button
              onClick={() => window.print()}
              className="px-2.5 py-1.5 rounded-lg text-xs font-semibold bg-navy-950 hover:bg-slate-800 border border-slate-700 text-slate-300 flex items-center gap-1 transition-all"
              title="Print this lesson"
            >
              <span>🖨️</span>
              <span className="hidden md:inline">Print</span>
            </button>

            {/* Status toggle */}
            <button
              onClick={() => onToggleStatus(dayItem.day_number)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all ${
                isCompleted
                  ? 'bg-emerald-950 text-emerald-300 border border-emerald-500/40 hover:bg-emerald-900'
                  : 'bg-amber-950/80 text-amber-300 border border-amber-500/40 hover:bg-amber-900'
              }`}
            >
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>{isCompleted ? 'Completed' : 'Mark Done'}</span>
            </button>

            {/* Nav prev/next */}
            <div className="flex items-center bg-navy-950 rounded-lg p-0.5 border border-slate-700">
              <button
                disabled={dayItem.day_number <= 1}
                onClick={() => onSelectDay(dayItem.day_number - 1)}
                className="p-1 rounded text-slate-400 hover:text-slate-100 disabled:opacity-30 disabled:hover:text-slate-400"
                title="Previous Day"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <span className="text-xs text-slate-400 px-1 font-mono">
                {dayItem.day_number}/{totalDays}
              </span>
              <button
                disabled={dayItem.day_number >= totalDays}
                onClick={() => onSelectDay(dayItem.day_number + 1)}
                className="p-1 rounded text-slate-400 hover:text-slate-100 disabled:opacity-30 disabled:hover:text-slate-400"
                title="Next Day"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>

            {/* Close */}
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-slate-400 hover:text-slate-100 hover:bg-slate-800 transition-colors ml-1"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Tab Navigation Bar */}
        <div className="px-6 bg-navy-950/70 border-b border-gold-500/10 flex items-center gap-1 overflow-x-auto scrollbar-none py-2">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition-all ${
                  isActive
                    ? 'bg-burgundy-900 text-gold-300 border border-gold-500/40 shadow'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-navy-900/50'
                }`}
              >
                <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-gold-400' : 'text-slate-400'}`} />
                <span>{tab.label}</span>
                {tab.badge !== undefined && (
                  <span className="px-1.5 py-0.2 rounded-full text-[10px] bg-navy-950 text-gold-300 border border-slate-700">
                    {tab.badge}
                  </span>
                )}
              </button>
            );
          })}
        </div>

        {/* Modal Scrollable Body */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6 text-slate-200 text-sm leading-relaxed">
          
          {/* TAB 1: NOTES */}
          {activeTab === 'notes' && (
            <div className="space-y-6">
              
              {/* Meta Card */}
              <div className="p-4 rounded-xl bg-navy-950/70 border border-gold-500/20 grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                <div>
                  <h4 className="text-gold-400 font-bold uppercase tracking-wider mb-1">Prerequisites</h4>
                  <p className="text-slate-300">{dayData.prerequisites || 'Basic Linear Algebra, Calculus, Python'}</p>
                </div>
                <div>
                  <h4 className="text-gold-400 font-bold uppercase tracking-wider mb-1">Textbook Reference</h4>
                  <p className="text-slate-300">{dayData.textbook_ref || 'Goodfellow et al. Deep Learning, MIT Press'}</p>
                </div>
                <div className="md:col-span-2">
                  <h4 className="text-gold-400 font-bold uppercase tracking-wider mb-1">Key Terminology</h4>
                  <div className="flex flex-wrap gap-1.5 mt-1">
                    {(dayData.key_terminology || ['Artificial Neuron', 'Feed-forward', 'Activation', 'Loss Function']).map((term: string, idx: number) => (
                      <span key={idx} className="px-2 py-0.5 rounded-md bg-navy-900 border border-slate-700 text-slate-200 text-[11px]">
                        {term}
                      </span>
                    ))}
                  </div>
                </div>
              </div>

              {/* Dynamic Note Sections */}
              {Object.entries(notes).map(([secKey, secVal]: [string, any]) => {
                const title = secKey.replace(/_/g, ' ').toUpperCase();
                const isExam = secKey.includes('exam');

                return (
                  <div 
                    key={secKey}
                    className={`p-5 rounded-xl border ${
                      isExam 
                        ? 'bg-burgundy-950/30 border-gold-500/40 text-gold-100' 
                        : 'bg-navy-950/50 border-slate-800 text-slate-200'
                    }`}
                  >
                    <h3 className={`text-sm font-bold uppercase tracking-wide flex items-center gap-2 mb-3 ${
                      isExam ? 'text-gold-300' : 'text-slate-100'
                    }`}>
                      {isExam ? <Award className="w-4 h-4 text-gold-400" /> : <BookOpen className="w-4 h-4 text-amber-400" />}
                      <span>{title}</span>
                    </h3>

                    <div className="text-xs md:text-sm text-slate-300 leading-relaxed whitespace-pre-line space-y-2 font-sans">
                      {typeof secVal === 'string' ? (
                        <p>{secVal}</p>
                      ) : Array.isArray(secVal) ? (
                        <ul className="list-disc list-inside space-y-1">
                          {secVal.map((item, idx) => (
                            <li key={idx}>{typeof item === 'string' ? item : JSON.stringify(item)}</li>
                          ))}
                        </ul>
                      ) : (
                        <pre className="p-3 bg-navy-950 rounded-lg text-xs font-mono text-slate-300 overflow-x-auto">
                          {JSON.stringify(secVal, null, 2)}
                        </pre>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* TAB 2: CLASSROOM SCRIPT */}
          {activeTab === 'script' && (
            <div className="space-y-6">
              <div className="p-4 rounded-xl bg-navy-950/70 border border-gold-500/20">
                <h3 className="text-sm font-bold text-gold-300 mb-1 flex items-center gap-2">
                  <Clock className="w-4 h-4 text-gold-400" />
                  <span>60-Minute Pedagogical Lesson Delivery Plan</span>
                </h3>
                <p className="text-xs text-slate-400">
                  Follow this structured timeline designed for KTU engineering classrooms to balance theoretical rigor, mathematical proofs, live demonstrations, and active student verification.
                </p>
              </div>

              <div className="relative border-l-2 border-gold-500/30 ml-4 pl-6 space-y-6">
                {Object.entries(script).map(([timeSlot, content]: [string, any], idx) => {
                  return (
                    <div key={timeSlot} className="relative group">
                      {/* Timeline dot */}
                      <div className="absolute -left-[31px] top-1 w-4 h-4 rounded-full bg-burgundy-800 border-2 border-gold-400 shadow" />
                      
                      <div className="p-4 rounded-xl bg-navy-950/60 border border-slate-800 hover:border-gold-500/30 transition-all">
                        <div className="flex items-center justify-between gap-2 mb-2">
                          <span className="px-2 py-0.5 rounded text-xs font-bold font-mono bg-burgundy-900/90 text-gold-300 border border-gold-500/30">
                            Minute {timeSlot}
                          </span>
                          <span className="text-[11px] text-slate-400 uppercase font-semibold">
                            {idx === 0 ? 'Recap & Context' : idx === 1 ? 'Intuitive Foundation' : idx === 2 ? 'Deep Mathematical Board Work' : idx === 3 ? 'Worked Problem Calculation' : idx === 4 ? 'Demonstration & Application' : 'Viva Check & Assessment'}
                          </span>
                        </div>
                        <p className="text-xs md:text-sm text-slate-300 leading-relaxed">
                          {typeof content === 'string' ? content : JSON.stringify(content)}
                        </p>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* TAB 3: SOLVED NUMERICALS */}
          {activeTab === 'numericals' && (
            <div className="space-y-6">
              {numericals.map((prob: any, idx: number) => (
                <div key={idx} className="p-5 rounded-xl bg-navy-950/70 border border-gold-500/30 space-y-4">
                  <div className="flex items-center justify-between gap-2 border-b border-slate-800 pb-3">
                    <div className="flex items-center gap-2">
                      <span className="px-2.5 py-0.5 rounded-full text-xs font-bold uppercase bg-burgundy-900 text-gold-300 border border-gold-500/30">
                        {prob.type || `Problem ${idx + 1}`}
                      </span>
                      <h3 className="text-sm font-bold text-slate-100">{prob.title}</h3>
                    </div>
                    <span className="text-xs text-gold-400 font-mono">KTU Format</span>
                  </div>

                  <div>
                    <h4 className="text-xs font-bold text-gold-300 uppercase tracking-wider mb-1">Problem Statement</h4>
                    <p className="text-xs md:text-sm text-slate-200 bg-navy-900/90 p-3 rounded-lg border border-slate-800">
                      {prob.statement}
                    </p>
                  </div>

                  {prob.given && (
                    <div>
                      <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Given Parameters</h4>
                      <code className="text-xs font-mono text-gold-200 bg-navy-950 px-2.5 py-1 rounded block border border-slate-800">
                        {prob.given}
                      </code>
                    </div>
                  )}

                  {prob.formulas && (
                    <div>
                      <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Governing Formulas</h4>
                      <code className="text-xs font-mono text-emerald-300 bg-navy-950 px-2.5 py-1 rounded block border border-slate-800">
                        {prob.formulas}
                      </code>
                    </div>
                  )}

                  {prob.steps && (
                    <div>
                      <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Step-by-Step Derivation &amp; Calculation</h4>
                      <div className="space-y-1.5">
                        {prob.steps.map((st: string, sIdx: number) => (
                          <div key={sIdx} className="text-xs text-slate-300 bg-navy-950/50 p-2 rounded border border-slate-800/80">
                            {st}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {prob.answer && (
                    <div className="p-3 rounded-lg bg-emerald-950/30 border border-emerald-500/40 text-emerald-200 text-xs flex items-start gap-2">
                      <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                      <div>
                        <strong>Final Evaluated Answer:</strong> {prob.answer}
                      </div>
                    </div>
                  )}

                  {prob.interpretation && (
                    <div className="text-xs text-slate-400 italic">
                      <strong className="text-slate-300 not-italic">Pedagogical Interpretation:</strong> {prob.interpretation}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* TAB 4: PYTHON DEMO */}
          {activeTab === 'demo' && (
            <div className="space-y-4">
              <div className="p-4 rounded-xl bg-navy-950/70 border border-gold-500/20 flex items-center justify-between gap-4">
                <div>
                  <h3 className="text-sm font-bold text-gold-300 flex items-center gap-2">
                    <Code2 className="w-4 h-4 text-gold-400" />
                    <span>{demo.title || 'Executable Python Numerical Demonstration'}</span>
                  </h3>
                  <p className="text-xs text-slate-400 mt-1">
                    {demo.description || 'Google Colab & Jupyter compatible reference code for interactive lecture demonstration.'}
                  </p>
                </div>

                <button
                  onClick={handleCopyCode}
                  className="px-3 py-1.5 bg-burgundy-900 hover:bg-burgundy-800 border border-gold-500/40 rounded-lg text-xs font-semibold text-gold-200 flex items-center gap-1.5 transition-all shadow shrink-0"
                >
                  {copiedCode ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5 text-gold-400" />}
                  <span>{copiedCode ? 'Copied!' : 'Copy Code'}</span>
                </button>
              </div>

              {/* Code viewer */}
              <div className="rounded-xl border border-slate-800 bg-navy-950 overflow-hidden shadow-inner">
                <div className="px-4 py-2 bg-navy-900 border-b border-slate-800 flex items-center justify-between text-xs text-slate-400">
                  <span className="font-mono">demo.py (Pure NumPy &amp; Matplotlib)</span>
                  <span>Python 3.10+ / Colab</span>
                </div>
                <pre className="p-4 text-xs font-mono text-emerald-300 overflow-x-auto leading-relaxed max-h-96">
                  {demo.code || '# Demonstration code loaded from daily package\nimport numpy as np\nprint("Demo initialized")'}
                </pre>
              </div>
            </div>
          )}

          {/* TAB 5: INDUSTRY CASE */}
          {activeTab === 'application' && (
            <div className="space-y-6">
              <div className="p-5 rounded-xl bg-navy-950/70 border border-gold-500/30 space-y-4">
                <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
                  <Globe className="w-5 h-5 text-gold-400" />
                  <div>
                    <span className="text-[10px] font-bold uppercase tracking-wider text-gold-400">
                      Domain: {app.domain || 'Industrial Artificial Intelligence'}
                    </span>
                    <h3 className="text-base font-bold text-slate-100">{app.title || 'Industrial Deployment Case'}</h3>
                  </div>
                </div>

                <div>
                  <h4 className="text-xs font-bold text-gold-300 uppercase tracking-wider mb-1">Production Workflow</h4>
                  <p className="text-xs md:text-sm text-slate-300 leading-relaxed bg-navy-900/60 p-3 rounded-lg border border-slate-800">
                    {app.workflow || 'End-to-end data processing, feature scaling, model inference, and continuous latency monitoring.'}
                  </p>
                </div>

                <div>
                  <h4 className="text-xs font-bold text-emerald-400 uppercase tracking-wider mb-1">Industrial Benefit &amp; ROI</h4>
                  <p className="text-xs md:text-sm text-slate-300 leading-relaxed bg-emerald-950/20 p-3 rounded-lg border border-emerald-500/20">
                    {app.benefit || 'Eliminates manual heuristic rules, achieving 99.4% precision with sub-millisecond inference times.'}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* TAB 6: VIVA VOCE */}
          {activeTab === 'viva' && (
            <div className="space-y-3">
              <div className="p-4 rounded-xl bg-navy-950/70 border border-gold-500/20 mb-4">
                <h3 className="text-sm font-bold text-gold-300 flex items-center gap-2">
                  <HelpCircle className="w-4 h-4 text-gold-400" />
                  <span>Curated Viva Voce &amp; Technical Interview Question Bank (10 Questions)</span>
                </h3>
                <p className="text-xs text-slate-400">
                  Ideal for oral examinations, series test vivas, and lab assessments.
                </p>
              </div>

              {viva.map((item: any, idx: number) => (
                <div key={idx} className="p-3.5 rounded-xl bg-navy-950/50 border border-slate-800 space-y-1.5 hover:border-gold-500/30 transition-all">
                  <div className="flex items-start gap-2">
                    <span className="px-2 py-0.5 rounded text-[11px] font-bold bg-burgundy-900 text-gold-300 border border-gold-500/30 shrink-0">
                      Q{idx + 1}
                    </span>
                    <h4 className="text-xs md:text-sm font-semibold text-slate-200">
                      {item.q}
                    </h4>
                  </div>
                  <div className="text-xs text-emerald-300/90 pl-8 leading-relaxed font-sans">
                    <strong className="text-slate-400 font-medium">Model Answer:</strong> {item.a}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* TAB 7: QUIZ & ASSESSMENT */}
          {activeTab === 'quiz' && (
            <div className="space-y-6">
              
              {/* MCQs */}
              <div className="space-y-4">
                <h3 className="text-sm font-bold text-gold-300 uppercase tracking-wider flex items-center gap-2">
                  <CheckSquare className="w-4 h-4 text-gold-400" />
                  <span>Objective Multiple Choice Quiz</span>
                </h3>

                {(assessment.mcqs || []).map((mcq: any, qIdx: number) => {
                  const selectedOpt = selectedQuizAnswers[qIdx];
                  const hasAnswered = selectedOpt !== undefined;
                  const isCorrect = selectedOpt === mcq.ans;

                  return (
                    <div key={qIdx} className="p-4 rounded-xl bg-navy-950/70 border border-slate-800 space-y-3">
                      <div className="flex items-start gap-2">
                        <span className="px-2 py-0.5 rounded text-[11px] font-bold bg-navy-900 text-gold-300 border border-slate-700 shrink-0">
                          MCQ {qIdx + 1}
                        </span>
                        <h4 className="text-xs md:text-sm font-semibold text-slate-100">{mcq.q}</h4>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-2 pl-6">
                        {(mcq.options || []).map((opt: string, oIdx: number) => {
                          const isSelected = selectedOpt === opt;
                          const isAnswerOption = opt === mcq.ans;

                          let btnStyle = 'bg-navy-900/80 text-slate-300 border-slate-800 hover:border-gold-500/40';
                          if (hasAnswered) {
                            if (isAnswerOption) {
                              btnStyle = 'bg-emerald-950 text-emerald-300 border-emerald-500';
                            } else if (isSelected && !isCorrect) {
                              btnStyle = 'bg-red-950 text-red-300 border-red-500';
                            }
                          }

                          return (
                            <button
                              key={oIdx}
                              onClick={() => handleQuizSelect(qIdx, opt)}
                              className={`p-2.5 rounded-lg border text-xs text-left transition-all ${btnStyle}`}
                            >
                              <span className="font-bold mr-2 text-gold-400">{String.fromCharCode(65 + oIdx)}.</span>
                              <span>{opt}</span>
                            </button>
                          );
                        })}
                      </div>

                      {hasAnswered && (
                        <div className={`text-xs pl-6 pt-1 font-semibold ${isCorrect ? 'text-emerald-400' : 'text-amber-400'}`}>
                          {isCorrect ? 'Correct! ' : `Incorrect. Correct answer is: ${mcq.ans}`}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

              {/* Short & Application Questions */}
              {assessment.short_questions && (
                <div className="p-4 rounded-xl bg-navy-950/50 border border-slate-800 space-y-3">
                  <h4 className="text-xs font-bold text-gold-300 uppercase tracking-wider">Descriptive Questions</h4>
                  <ul className="list-disc list-inside text-xs text-slate-300 space-y-1">
                    {assessment.short_questions.map((sq: string, idx: number) => (
                      <li key={idx}>{sq}</li>
                    ))}
                  </ul>
                </div>
              )}

              {assessment.homework && (
                <div className="p-4 rounded-xl bg-burgundy-950/20 border border-gold-500/30 text-xs">
                  <h4 className="font-bold text-gold-300 uppercase tracking-wider mb-1">Homework Assignment</h4>
                  <p className="text-slate-300">{assessment.homework}</p>
                </div>
              )}
            </div>
          )}

          {/* TAB 8: ASSETS & DOWNLOADS */}
          {activeTab === 'files' && (
            <div className="space-y-4">
              
              {/* Featured Downloads Bar */}
              <div className="p-5 rounded-xl bg-gradient-to-r from-navy-950 via-burgundy-950/70 to-navy-950 border border-gold-500/40 space-y-3">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-sm font-bold text-gold-300 flex items-center gap-2">
                      <Download className="w-4 h-4 text-gold-400" />
                      <span>Printable Handwritten PDF &amp; Slide Downloads</span>
                    </h3>
                    <p className="text-xs text-slate-300 mt-0.5">
                      Authentic handwritten Kalam typography formatted for A4 printing and double-sided study booklets.
                    </p>
                  </div>
                  <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-950 text-emerald-300 border border-emerald-500/40">
                    Ready to Print
                  </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2 pt-1">
                  <a
                    href={`/downloads/CST414_Module_${dayItem.module}_Handwritten_Notes.pdf`}
                    download={`CST414_Module_${dayItem.module}_Handwritten_Notes.pdf`}
                    className="p-2.5 rounded-lg bg-burgundy-900/80 hover:bg-burgundy-800 border border-gold-500/40 text-gold-200 text-xs font-semibold flex items-center justify-between transition-all shadow"
                  >
                    <span>📄 Module {dayItem.module} Handwritten PDF</span>
                    <Download className="w-3.5 h-3.5 text-gold-400" />
                  </a>

                  <a
                    href="/downloads/CST414_Handwritten_Lecture_Slides_Handout.pdf"
                    download="CST414_Handwritten_Lecture_Slides_Handout.pdf"
                    className="p-2.5 rounded-lg bg-navy-900 hover:bg-slate-800 border border-slate-700 text-slate-200 text-xs font-semibold flex items-center justify-between transition-all"
                  >
                    <span>📑 Slides Handout (PDF)</span>
                    <Download className="w-3.5 h-3.5 text-gold-400" />
                  </a>

                  <a
                    href={`/downloads/Module_${dayItem.module}_Master_Presentation.pptx`}
                    download={`Module_${dayItem.module}_Master_Presentation.pptx`}
                    className="p-2.5 rounded-lg bg-navy-900 hover:bg-slate-800 border border-slate-700 text-slate-200 text-xs font-semibold flex items-center justify-between transition-all"
                  >
                    <span>📊 Module {dayItem.module} Slides (.pptx)</span>
                    <Download className="w-3.5 h-3.5 text-gold-400" />
                  </a>

                  <button
                    onClick={() => window.print()}
                    className="p-2.5 rounded-lg bg-navy-900 hover:bg-slate-800 border border-slate-700 text-gold-300 text-xs font-semibold flex items-center justify-between transition-all"
                  >
                    <span>🖨️ Print Lesson (PDF)</span>
                    <span className="text-[10px] text-slate-400">Ctrl+P</span>
                  </button>
                </div>
              </div>

              <div className="p-4 rounded-xl bg-navy-950/70 border border-gold-500/20">
                <h3 className="text-sm font-bold text-gold-300 mb-1 flex items-center gap-2">
                  <FileText className="w-4 h-4 text-gold-400" />
                  <span>All 9 Generated Daily Files for Day #{dayItem.day_number.toString().padStart(2, '0')}</span>
                </h3>
                <p className="text-xs text-slate-400">
                  Stored in local repository at: <code className="text-gold-300">DEEP_LEARNING_CST414/03_DAILY_CLASS_CONTENT/.../DAY_{dayItem.day_number.toString().padStart(2, '0')}/</code>
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                {[
                  { name: 'Daily_Lesson_Notes.docx', desc: 'Complete KTU syllabus notes with math derivations', ext: 'DOCX', dl: `/downloads/CST414_Module_${dayItem.module}_Handwritten_Notes.pdf` },
                  { name: 'Daily_Lecture_Slides.pptx', desc: '10-slide faculty presentation deck', ext: 'PPTX', dl: `/downloads/Module_${dayItem.module}_Master_Presentation.pptx` },
                  { name: 'Teaching_Script.docx', desc: 'Pedagogical delivery script with class checkpoints', ext: 'DOCX', dl: null },
                  { name: 'Python_Demo.ipynb', desc: 'Executable Jupyter / Colab demo notebook', ext: 'IPYNB', dl: null },
                  { name: 'Numerical_Problems_With_Solutions.docx', desc: '2 solved numericals with step-by-step arithmetic', ext: 'DOCX', dl: `/downloads/CST414_Module_${dayItem.module}_Handwritten_Notes.pdf` },
                  { name: 'Real_World_Application.docx', desc: 'Production case study & pipeline architecture', ext: 'DOCX', dl: null },
                  { name: 'Viva_Questions_With_Answers.docx', desc: '10 curated oral questions with model answers', ext: 'DOCX', dl: `/downloads/CST414_Module_${dayItem.module}_Handwritten_Notes.pdf` },
                  { name: 'Class_Quiz.docx', desc: 'Objective MCQs and descriptive homework questions', ext: 'DOCX', dl: null },
                  { name: 'Reference_Materials.md', desc: 'Official textbook citations and further reading', ext: 'MD', dl: null },
                ].map((file, fIdx) => (
                  <div key={fIdx} className="p-3.5 rounded-xl bg-navy-950/60 border border-slate-800 hover:border-gold-500/40 transition-all flex flex-col justify-between">
                    <div>
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-navy-900 text-gold-300 border border-slate-700">
                          {file.ext}
                        </span>
                        <FileText className="w-4 h-4 text-slate-400" />
                      </div>
                      <h4 className="text-xs font-bold text-slate-200 truncate">{file.name}</h4>
                      <p className="text-[11px] text-slate-400 mt-1 line-clamp-2">{file.desc}</p>
                    </div>

                    {file.dl ? (
                      <a
                        href={file.dl}
                        download
                        className="mt-3 w-full py-1.5 bg-navy-900 hover:bg-burgundy-900 border border-slate-700 hover:border-gold-500/40 rounded-lg text-[11px] font-semibold text-gold-300 transition-all flex items-center justify-center gap-1"
                      >
                        <Download className="w-3 h-3" />
                        <span>Download Asset</span>
                      </a>
                    ) : (
                      <button
                        onClick={() => alert(`Available in repository at: DEEP_LEARNING_CST414/03_DAILY_CLASS_CONTENT/.../DAY_${dayItem.day_number.toString().padStart(2, '0')}/${file.name}`)}
                        className="mt-3 w-full py-1.5 bg-navy-900/60 hover:bg-navy-800 border border-slate-800 rounded-lg text-[11px] font-medium text-slate-400 transition-all flex items-center justify-center gap-1"
                      >
                        <span>Local Repo File</span>
                      </button>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

        </div>

        {/* Modal Footer */}
        <div className="px-6 py-3 bg-navy-950 border-t border-gold-500/20 flex items-center justify-between text-xs text-slate-400">
          <div>
            Faculty: <strong className="text-gold-200">Jane Rose</strong> | KTU CST414 Deep Learning
          </div>
          <button
            onClick={onClose}
            className="px-4 py-1.5 bg-navy-900 hover:bg-slate-800 border border-slate-700 rounded-lg text-xs font-semibold text-slate-200 transition-colors"
          >
            Close Modal
          </button>
        </div>

      </div>
    </div>
  );
};

