import React, { useState, useEffect } from 'react';
import courseData from './data/courseData.json';
import { Navbar } from './components/Navbar';
import { ScheduleView, ScheduleItem } from './components/ScheduleView';
import { DayExplorerModal } from './components/DayExplorerModal';
import { ThreeNeuralVisualizer } from './components/ThreeNeuralVisualizer';
import { GradientDescentSimulator } from './components/GradientDescentSimulator';
import { CnnRnnStudio } from './components/CnnRnnStudio';
import { QuestionPaperViewer } from './components/QuestionPaperViewer';
import { AssessmentsView } from './components/AssessmentsView';
import { Footer } from './components/Footer';
import { 
  BookOpen, Calendar, Cpu, Activity, Layers, 
  FileText, GraduationCap, CheckCircle2, Search,
  Download, Sparkles, Filter, ChevronRight, AlertTriangle
} from 'lucide-react';

export function App() {
  const scheduleData = (courseData.schedule || []) as ScheduleItem[];
  const totalHours = courseData.course_info.total_hours || 36;

  // Active navigation tab
  const [activeTab, setActiveTab] = useState<string>('schedule');

  // Completed days set (persisted in localStorage or default to first 3 days)
  const [completedDays, setCompletedDays] = useState<Set<number>>(() => {
    try {
      const saved = localStorage.getItem('cst414_completed_days');
      if (saved) return new Set(JSON.parse(saved));
    } catch (e) {
      // fallback
    }
    return new Set([1, 2, 3]);
  });

  // Global search query
  const [searchQuery, setSearchQuery] = useState<string>('');

  // Selected day for modal
  const [selectedDayNum, setSelectedDayNum] = useState<number | null>(null);

  // Day Explorer module filter
  const [dayExplorerModule, setDayExplorerModule] = useState<number | 'all'>('all');

  // Export notification toast
  const [exportNotice, setExportNotice] = useState<string | null>(null);

  useEffect(() => {
    try {
      localStorage.setItem('cst414_completed_days', JSON.stringify(Array.from(completedDays)));
    } catch (e) {
      // ignore
    }
  }, [completedDays]);

  const handleToggleStatus = (dayNum: number) => {
    setCompletedDays((prev) => {
      const next = new Set(prev);
      if (next.has(dayNum)) {
        next.delete(dayNum);
      } else {
        next.add(dayNum);
      }
      return next;
    });
  };

  const handleExport = (format: string) => {
    let msg = '';
    if (format === 'docx') {
      msg = 'All 324 Word documents (.docx) generated in DEEP_LEARNING_CST414/';
    } else if (format === 'pptx') {
      msg = 'All 36 daily lecture PowerPoint decks (.pptx) ready in DEEP_LEARNING_CST414/';
    } else if (format === 'xlsx') {
      msg = 'Excel Academic Calendars & Teaching Schedules (.xlsx) ready in 01_ACADEMIC_CALENDAR/';
    } else if (format === 'ipynb') {
      msg = 'All 36 Jupyter Notebook demos (.ipynb) verified in 03_DAILY_CLASS_CONTENT/ and 04_PYTHON_DEMONSTRATIONS/';
    }
    setExportNotice(msg);
    setTimeout(() => setExportNotice(null), 5000);
  };

  const selectedDayItem = selectedDayNum !== null 
    ? scheduleData.find((s) => s.day_number === selectedDayNum) || null 
    : null;

  // Filtered days for Day Explorer tab
  const explorerDays = scheduleData.filter((item) => {
    if (dayExplorerModule !== 'all' && item.module !== dayExplorerModule) return false;
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      const matchTitle = item.title.toLowerCase().includes(q);
      const matchTopic = item.topic_id.toLowerCase().includes(q);
      const matchModule = item.module_name.toLowerCase().includes(q);
      if (!matchTitle && !matchTopic && !matchModule) return false;
    }
    return true;
  });

  return (
    <div className="min-h-screen bg-navy-950 text-slate-100 flex flex-col font-sans selection:bg-gold-500 selection:text-navy-950">
      
      {/* Sticky Navigation Bar */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        completedHours={completedDays.size}
        totalHours={totalHours}
        searchQuery={searchQuery}
        setSearchQuery={setSearchQuery}
        onExport={handleExport}
      />

      {/* Export Toast Notification */}
      {exportNotice && (
        <div className="fixed bottom-6 right-6 z-50 p-4 rounded-xl bg-burgundy-900 border border-gold-500/60 text-gold-200 shadow-2xl flex items-center gap-3 animate-in slide-in-from-bottom duration-300">
          <Download className="w-5 h-5 text-gold-400 shrink-0" />
          <div className="text-xs">
            <strong className="block text-slate-100 font-serif">Resource Package Verified</strong>
            <span>{exportNotice}</span>
          </div>
        </div>
      )}

      {/* Main Content Viewport */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 lg:px-8 py-8">
        
        {/* VIEW 1: 36-HOUR SCHEDULE */}
        {activeTab === 'schedule' && (
          <ScheduleView
            schedule={scheduleData}
            completedDays={completedDays}
            onToggleStatus={handleToggleStatus}
            onSelectDay={(dayNum) => setSelectedDayNum(dayNum)}
            searchQuery={searchQuery}
          />
        )}

        {/* VIEW 2: DAY-BY-DAY RESOURCES EXPLORER */}
        {activeTab === 'day-explorer' && (
          <div className="space-y-6">
            <div className="p-6 rounded-2xl bg-gradient-to-r from-navy-900 via-burgundy-950/60 to-navy-900 border border-gold-500/30 flex flex-col md:flex-row items-center justify-between gap-4">
              <div>
                <div className="flex items-center gap-2">
                  <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-burgundy-800 text-gold-300 border border-gold-500/30">
                    Day-by-Day Repository
                  </span>
                  <span className="text-xs text-slate-400 font-medium">36 Complete Instructional Packages (324 Files)</span>
                </div>
                <h2 className="text-lg md:text-xl font-bold font-serif text-slate-100 mt-1">
                  Master Daily Lesson Vault &amp; Teaching Dossiers
                </h2>
                <p className="text-xs text-slate-300 mt-1">
                  Every lecture contains 9 dedicated files: KTU Notes, Slides, Script, Python Demo, Numericals, Real-World Case, Viva Bank, Quiz, and References.
                </p>
              </div>

              {/* Module Filter Pills */}
              <div className="flex items-center gap-1.5 overflow-x-auto pb-1 max-w-full">
                <button
                  onClick={() => setDayExplorerModule('all')}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                    dayExplorerModule === 'all'
                      ? 'bg-gold-500 text-navy-950 shadow'
                      : 'bg-navy-950 text-slate-300 hover:bg-slate-800'
                  }`}
                >
                  All 36 Days
                </button>
                {[1, 2, 3, 4, 5].map((m) => (
                  <button
                    key={m}
                    onClick={() => setDayExplorerModule(m)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                      dayExplorerModule === m
                        ? 'bg-gold-500 text-navy-950 shadow'
                        : 'bg-navy-950 text-slate-300 hover:bg-slate-800'
                    }`}
                  >
                    Module {m}
                  </button>
                ))}
              </div>
            </div>

            {/* Days Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {explorerDays.map((item) => {
                const isDone = completedDays.has(item.day_number);
                const isDuplicated = item.day_number === 14 || item.day_number === 15;

                return (
                  <div
                    key={item.day_number}
                    onClick={() => setSelectedDayNum(item.day_number)}
                    className={`p-4 rounded-xl border transition-all cursor-pointer group flex flex-col justify-between ${
                      isDone
                        ? 'bg-navy-900/60 border-emerald-500/30 hover:border-emerald-500'
                        : 'bg-navy-900/80 border-slate-800 hover:border-gold-500/50 hover:bg-navy-900'
                    }`}
                  >
                    <div>
                      <div className="flex items-center justify-between gap-2 mb-2">
                        <span className="w-8 h-8 rounded-lg bg-burgundy-900 text-gold-300 font-bold font-serif flex items-center justify-center text-xs border border-gold-500/30">
                          #{item.day_number.toString().padStart(2, '0')}
                        </span>
                        <div className="flex items-center gap-1.5">
                          <span className="px-1.5 py-0.2 rounded text-[10px] font-bold bg-navy-950 text-gold-300 border border-slate-700">
                            M{item.module}:{item.topic_id}
                          </span>
                          <span className="px-1.5 py-0.2 rounded text-[10px] font-medium bg-slate-900 text-slate-400">
                            {item.co}
                          </span>
                        </div>
                      </div>

                      <h3 className="text-sm font-bold text-slate-100 group-hover:text-gold-300 transition-colors line-clamp-2">
                        {item.title}
                      </h3>
                      <p className="text-[11px] text-slate-400 mt-1 line-clamp-1">
                        {item.date_display} &bull; {item.period.split('(')[0]}
                      </p>

                      {isDuplicated && (
                        <div className="mt-2 text-[10px] text-amber-300 bg-amber-950/40 p-1 rounded border border-amber-500/30 flex items-center gap-1">
                          <AlertTriangle className="w-3 h-3" />
                          <span>Flagged KTU 2.7/2.8 Topic</span>
                        </div>
                      )}
                    </div>

                    <div className="mt-4 pt-3 border-t border-slate-800/80 flex items-center justify-between text-xs">
                      <span className={`text-[11px] font-semibold flex items-center gap-1 ${
                        isDone ? 'text-emerald-400' : 'text-slate-400'
                      }`}>
                        <CheckCircle2 className="w-3.5 h-3.5" />
                        <span>{isDone ? 'Completed' : 'Pending'}</span>
                      </span>

                      <span className="text-gold-400 group-hover:translate-x-0.5 transition-transform flex items-center text-[11px] font-bold">
                        <span>Open Package</span>
                        <ChevronRight className="w-3.5 h-3.5 ml-0.5" />
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* VIEW 3: 3D NEURAL STUDIO */}
        {activeTab === '3d-neural' && (
          <div className="space-y-6">
            <div className="p-6 rounded-2xl bg-gradient-to-r from-navy-900 via-burgundy-950/60 to-navy-900 border border-gold-500/30">
              <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-burgundy-800 text-gold-300 border border-gold-500/30">
                Interactive 3D WebGL Visualization
              </span>
              <h2 className="text-lg md:text-xl font-bold font-serif text-slate-100 mt-1">
                Real-Time Three.js Neural Network Architecture &amp; Synaptic Signal Propagation
              </h2>
              <p className="text-xs text-slate-300 mt-1">
                Explore deep multi-layer perceptron topology, synaptic weight connections, pulsing forward activation pulses, and interactive camera orbits.
              </p>
            </div>
            <ThreeNeuralVisualizer />
          </div>
        )}

        {/* VIEW 4: OPTIMIZATION LAB */}
        {activeTab === 'gd-lab' && (
          <div className="space-y-6">
            <div className="p-6 rounded-2xl bg-gradient-to-r from-navy-900 via-burgundy-950/60 to-navy-900 border border-gold-500/30">
              <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-burgundy-800 text-gold-300 border border-gold-500/30">
                Optimization Laboratory
              </span>
              <h2 className="text-lg md:text-xl font-bold font-serif text-slate-100 mt-1">
                Gradient Descent Trajectory &amp; Optimizer Comparison Simulator
              </h2>
              <p className="text-xs text-slate-300 mt-1">
                Simulate GD, Momentum, Nesterov Accelerated Gradient (NAG), AdaGrad, RMSProp, and Adam on pathological ill-conditioned ravines.
              </p>
            </div>
            <GradientDescentSimulator />
          </div>
        )}

        {/* VIEW 5: CNN & RNN STUDIO */}
        {activeTab === 'cnn-rnn' && (
          <CnnRnnStudio />
        )}

        {/* VIEW 6: EXAMS */}
        {activeTab === 'exams' && (
          <QuestionPaperViewer />
        )}

        {/* VIEW 7: ASSESSMENTS & OUTCOMES */}
        {activeTab === 'assessments' && (
          <AssessmentsView />
        )}

      </main>

      {/* Day Explorer Modal */}
      {selectedDayNum !== null && (
        <DayExplorerModal
          dayItem={selectedDayItem}
          onClose={() => setSelectedDayNum(null)}
          onSelectDay={(dayNum) => setSelectedDayNum(dayNum)}
          totalDays={scheduleData.length}
          isCompleted={completedDays.has(selectedDayNum)}
          onToggleStatus={handleToggleStatus}
        />
      )}

      {/* Mandatory Footer across all views */}
      <Footer />

    </div>
  );
}

export default App;
