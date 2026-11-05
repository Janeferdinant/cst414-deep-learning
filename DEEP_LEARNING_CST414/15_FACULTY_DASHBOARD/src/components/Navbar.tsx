import React from 'react';
import { 
  BookOpen, Calendar, Cpu, Activity, Layers, 
  FileText, CheckCircle2, AlertTriangle, Download, 
  Search, ShieldCheck, GraduationCap
} from 'lucide-react';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  completedHours: number;
  totalHours: number;
  searchQuery: string;
  setSearchQuery: (q: string) => void;
  onExport: (format: string) => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  completedHours,
  totalHours,
  searchQuery,
  setSearchQuery,
  onExport
}) => {
  const navItems = [
    { id: 'schedule', label: '36-Hour Schedule', icon: Calendar },
    { id: 'day-explorer', label: 'Day-by-Day Resources', icon: BookOpen },
    { id: '3d-neural', label: '3D Neural Studio', icon: Cpu },
    { id: 'gd-lab', label: 'Optimization Lab', icon: Activity },
    { id: 'cnn-rnn', label: 'CNN & RNN Studio', icon: Layers },
    { id: 'exams', label: 'Solved KTU Papers', icon: FileText },
    { id: 'assessments', label: 'Assignments & COs', icon: GraduationCap },
  ];

  const progressPct = Math.round((completedHours / totalHours) * 100);

  return (
    <header className="sticky top-0 z-50 glass-panel border-b border-gold-500/20 px-4 lg:px-8 py-3">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
        
        {/* Brand & Faculty Info */}
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-burgundy-700 via-burgundy-900 to-navy-950 border border-gold-500/40 flex items-center justify-center shadow-lg shadow-burgundy-900/50">
            <Cpu className="w-6 h-6 text-gold-400 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 rounded text-[10px] font-bold tracking-wider uppercase bg-burgundy-800/80 text-gold-300 border border-gold-500/30">
                KTU CST414
              </span>
              <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-amber-500/20 text-amber-300 border border-amber-500/30 flex items-center gap-1">
                <AlertTriangle className="w-3 h-3" /> Provisional Schedule
              </span>
            </div>
            <h1 className="text-lg font-serif font-bold gold-gradient-text tracking-wide">
              Deep Learning <span className="text-xs font-sans font-normal text-slate-400">| Scheme 2019</span>
            </h1>
            <p className="text-xs text-slate-300 font-medium">
              Faculty: <strong className="text-gold-200">Jane Rose</strong> | Dept. of Computer Science & Engineering
            </p>
          </div>
        </div>

        {/* Center Progress & Stats */}
        <div className="flex items-center gap-4 bg-navy-900/80 border border-gold-500/20 rounded-xl px-4 py-2">
          <div className="text-right">
            <div className="text-xs text-slate-400">Instructional Progress</div>
            <div className="text-sm font-bold text-slate-200">
              <span className="text-gold-400">{completedHours}</span> / {totalHours} Hours ({progressPct}%)
            </div>
          </div>
          <div className="w-24 bg-navy-950 h-2.5 rounded-full overflow-hidden border border-slate-700">
            <div 
              className="bg-gradient-to-r from-burgundy-600 via-gold-500 to-amber-300 h-full rounded-full transition-all duration-500"
              style={{ width: `${progressPct}%` }}
            />
          </div>
        </div>

        {/* Export & Global Search */}
        <div className="flex items-center gap-2">
          <div className="relative">
            <Search className="w-4 h-4 absolute left-3 top-2.5 text-slate-400" />
            <input 
              type="text"
              placeholder="Search topics, equations..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9 pr-3 py-1.5 bg-navy-950/80 border border-slate-700 rounded-lg text-xs text-slate-200 focus:outline-none focus:border-gold-500/50 w-44 lg:w-56"
            />
          </div>

          <div className="dropdown relative group">
            <button className="flex items-center gap-1.5 px-3 py-1.5 bg-gradient-to-r from-burgundy-800 to-burgundy-900 hover:from-burgundy-700 hover:to-burgundy-800 border border-gold-500/40 rounded-lg text-xs font-semibold text-gold-200 shadow-md transition-all">
              <Download className="w-3.5 h-3.5" />
              <span>Downloads &amp; Print</span>
            </button>
            <div className="hidden group-hover:block absolute right-0 mt-1 w-64 bg-navy-900 border border-gold-500/40 rounded-xl shadow-2xl py-2 z-50 text-xs">
              <div className="px-3 py-1 text-[10px] font-bold uppercase tracking-wider text-gold-400 border-b border-slate-800 mb-1">
                Printable Handwritten PDFs
              </div>
              <a 
                href="/downloads/CST414_Deep_Learning_Master_Handwritten_Notes.pdf" 
                download="CST414_Deep_Learning_Master_Handwritten_Notes.pdf"
                className="w-full text-left px-3 py-1.5 text-slate-200 hover:bg-burgundy-900/60 hover:text-gold-300 flex items-center justify-between transition-colors"
              >
                <span className="font-semibold text-gold-200">📄 Master Handwritten Notes</span>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-500/40">36 Days</span>
              </a>
              <a 
                href="/downloads/CST414_Handwritten_Lecture_Slides_Handout.pdf" 
                download="CST414_Handwritten_Lecture_Slides_Handout.pdf"
                className="w-full text-left px-3 py-1.5 text-slate-200 hover:bg-burgundy-900/60 hover:text-gold-300 flex items-center justify-between transition-colors"
              >
                <span className="font-semibold text-gold-200">📑 Slides Handout (2/Page)</span>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-navy-950 text-gold-300 border border-slate-700">PDF Print</span>
              </a>

              <div className="px-3 py-1 text-[10px] font-bold uppercase tracking-wider text-slate-400 border-t border-b border-slate-800 my-1">
                Module-Wise Handwritten Booklets
              </div>
              {[1, 2, 3, 4, 5].map((m) => (
                <a
                  key={m}
                  href={`/downloads/CST414_Module_${m}_Handwritten_Notes.pdf`}
                  download={`CST414_Module_${m}_Handwritten_Notes.pdf`}
                  className="w-full text-left px-3 py-1 text-slate-300 hover:bg-burgundy-900/40 hover:text-gold-200 flex items-center justify-between transition-colors"
                >
                  <span>Module {m} Handwritten PDF</span>
                  <span className="text-[10px] text-slate-400">PDF</span>
                </a>
              ))}

              <div className="px-3 py-1 text-[10px] font-bold uppercase tracking-wider text-slate-400 border-t border-b border-slate-800 my-1">
                PowerPoint Slide Decks
              </div>
              <a 
                href="/downloads/CST414_All_Lecture_Slides_PPTX.zip" 
                download="CST414_All_Lecture_Slides_PPTX.zip"
                className="w-full text-left px-3 py-1.5 text-slate-200 hover:bg-burgundy-900/60 hover:text-gold-300 flex items-center justify-between transition-colors"
              >
                <span>📊 All 5 Master PPTX Slides</span>
                <span className="text-[10px] text-slate-400">ZIP</span>
              </a>

              <div className="border-t border-slate-800 pt-1 mt-1">
                <button
                  onClick={() => window.print()}
                  className="w-full text-left px-3 py-1.5 text-slate-200 hover:bg-slate-800 text-gold-300 flex items-center justify-between font-semibold"
                >
                  <span>🖨️ Print / Save View to PDF</span>
                  <span className="text-[10px] text-slate-400">Ctrl+P</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Pills Bar */}
      <div className="max-w-7xl mx-auto mt-3 pt-2 border-t border-slate-800 flex items-center gap-1 overflow-x-auto pb-1 scrollbar-none">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition-all duration-200 ${
                isActive
                  ? 'bg-gradient-to-r from-burgundy-700 to-burgundy-900 text-gold-200 border border-gold-500/50 shadow-md shadow-burgundy-950'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-navy-900/60 border border-transparent'
              }`}
            >
              <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-gold-400' : 'text-slate-400'}`} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </div>
    </header>
  );
};

