import React, { useState, useMemo } from 'react';
import { 
  Calendar, CheckCircle2, Clock, AlertTriangle, Search, 
  Filter, BookOpen, ChevronRight, Download, RefreshCw,
  Layers, ArrowUpDown, FileText, CheckCircle
} from 'lucide-react';

export interface ScheduleItem {
  day_number: number;
  date_str: string;
  date_display: string;
  day_name: string;
  period: string;
  module: number;
  module_name: string;
  topic_id: string;
  title: string;
  planned_hours: number;
  status: string;
  co: string;
  blooms: string;
  data?: any;
}

interface ScheduleViewProps {
  schedule: ScheduleItem[];
  completedDays: Set<number>;
  onToggleStatus: (dayNum: number) => void;
  onSelectDay: (dayNum: number) => void;
  searchQuery: string;
}

export const ScheduleView: React.FC<ScheduleViewProps> = ({
  schedule,
  completedDays,
  onToggleStatus,
  onSelectDay,
  searchQuery
}) => {
  const [selectedModule, setSelectedModule] = useState<number | 'all'>('all');
  const [statusFilter, setStatusFilter] = useState<'all' | 'completed' | 'pending'>('all');
  const [sortBy, setSortBy] = useState<'day' | 'date'>('day');

  // Filter and sort items
  const filteredSchedule = useMemo(() => {
    return schedule.filter((item) => {
      // Module filter
      if (selectedModule !== 'all' && item.module !== selectedModule) {
        return false;
      }
      // Status filter
      const isCompleted = completedDays.has(item.day_number);
      if (statusFilter === 'completed' && !isCompleted) return false;
      if (statusFilter === 'pending' && isCompleted) return false;

      // Search query filter
      if (searchQuery.trim()) {
        const query = searchQuery.toLowerCase();
        const matchesTitle = item.title.toLowerCase().includes(query);
        const matchesTopicId = item.topic_id.toLowerCase().includes(query);
        const matchesModule = item.module_name.toLowerCase().includes(query);
        const matchesCO = item.co.toLowerCase().includes(query);
        const matchesBlooms = item.blooms.toLowerCase().includes(query);
        if (!matchesTitle && !matchesTopicId && !matchesModule && !matchesCO && !matchesBlooms) {
          return false;
        }
      }

      return true;
    }).sort((a, b) => {
      if (sortBy === 'day') return a.day_number - b.day_number;
      return new Date(a.date_str).getTime() - new Date(b.date_str).getTime();
    });
  }, [schedule, selectedModule, statusFilter, searchQuery, sortBy, completedDays]);

  const totalHours = schedule.reduce((sum, item) => sum + item.planned_hours, 0);
  const completedHoursCount = Array.from(completedDays).length;
  const progressPct = Math.round((completedHoursCount / totalHours) * 100);

  const exportCSV = () => {
    const headers = ['Day', 'Date', 'Day of Week', 'Period', 'Module', 'Topic ID', 'Topic Title', 'Hours', 'CO', "Bloom's Taxonomy", 'Status'];
    const rows = schedule.map((item) => [
      item.day_number,
      item.date_display,
      item.day_name,
      `"${item.period}"`,
      item.module,
      item.topic_id,
      `"${item.title.replace(/"/g, '""')}"`,
      item.planned_hours,
      item.co,
      `"${item.blooms}"`,
      completedDays.has(item.day_number) ? 'Completed' : 'Pending'
    ]);

    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', 'CST414_Deep_Learning_36_Hour_Schedule.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="space-y-6">
      
      {/* Top Banner: Provisional Alert & Topic 2.7/2.8 Note */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="p-4 rounded-xl bg-amber-950/30 border border-amber-500/40 text-amber-200 text-xs flex items-start gap-3 shadow-lg shadow-amber-950/20">
          <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <h4 className="font-bold text-amber-300 uppercase tracking-wide">
              Provisional Institutional Schedule
            </h4>
            <p className="text-slate-300 leading-relaxed">
              Mapped across KTU S7 standard timeline (18-Aug-2025 to 26-Nov-2025, 3 lecture periods/week). When institutional college calendar and approved timetable are released, dates will be re-synchronized automatically.
            </p>
          </div>
        </div>

        <div className="p-4 rounded-xl bg-burgundy-950/40 border border-gold-500/30 text-gold-200 text-xs flex items-start gap-3 shadow-lg shadow-burgundy-950/30">
          <Layers className="w-5 h-5 text-gold-400 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <h4 className="font-bold text-gold-300 uppercase tracking-wide">
              Syllabus Duplication Flag (Topic 2.7 &amp; 2.8)
            </h4>
            <p className="text-slate-300 leading-relaxed">
              Official KTU syllabus lists identical titles for Topics 2.7 &amp; 2.8. Mapped: <strong>Day 14 (Topic 2.7)</strong> = Parameter Sharing, Tying &amp; Input Noise; <strong>Day 15 (Topic 2.8)</strong> = Ensemble Methods (Bagging, Boosting, Model Averaging).
            </p>
          </div>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="p-4 rounded-xl bg-navy-900/90 border border-gold-500/20 shadow-md">
          <div className="text-xs text-slate-400 font-medium">Total Lecture Hours</div>
          <div className="text-2xl font-bold font-serif text-slate-100 mt-1">36 <span className="text-xs font-sans text-gold-400 font-normal">Hours (100%)</span></div>
          <div className="text-[11px] text-slate-400 mt-1 flex items-center gap-1">
            <CheckCircle className="w-3 h-3 text-emerald-400" /> KTU Scheme 2019 Mandate
          </div>
        </div>

        <div className="p-4 rounded-xl bg-navy-900/90 border border-gold-500/20 shadow-md">
          <div className="text-xs text-slate-400 font-medium">Completed Lectures</div>
          <div className="text-2xl font-bold font-serif text-emerald-400 mt-1">
            {completedHoursCount} <span className="text-xs font-sans text-slate-400 font-normal">/ 36 Hrs</span>
          </div>
          <div className="text-[11px] text-slate-400 mt-1">
            {progressPct}% syllabus taught
          </div>
        </div>

        <div className="p-4 rounded-xl bg-navy-900/90 border border-gold-500/20 shadow-md">
          <div className="text-xs text-slate-400 font-medium">Remaining Lectures</div>
          <div className="text-2xl font-bold font-serif text-amber-300 mt-1">
            {totalHours - completedHoursCount} <span className="text-xs font-sans text-slate-400 font-normal">Hours</span>
          </div>
          <div className="text-[11px] text-slate-400 mt-1">
            Target End: 26-Nov-2025
          </div>
        </div>

        <div className="p-4 rounded-xl bg-navy-900/90 border border-gold-500/20 shadow-md">
          <div className="text-xs text-slate-400 font-medium">Course Coordinator</div>
          <div className="text-sm font-bold text-gold-300 mt-1 truncate">Jane Rose</div>
          <div className="text-[11px] text-slate-400 mt-0.5 truncate">
            Dept. of Computer Science &amp; Engg.
          </div>
        </div>
      </div>

      {/* Filter and Control Toolbar */}
      <div className="p-4 rounded-xl bg-navy-900/80 border border-gold-500/20 flex flex-wrap items-center justify-between gap-4">
        
        {/* Module Selector Pills */}
        <div className="flex items-center gap-1.5 overflow-x-auto pb-1 max-w-full">
          <span className="text-xs font-bold text-gold-300/80 uppercase mr-1 flex items-center gap-1">
            <Filter className="w-3.5 h-3.5" /> Module:
          </span>
          <button
            onClick={() => setSelectedModule('all')}
            className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all ${
              selectedModule === 'all'
                ? 'bg-gold-500 text-navy-950 shadow-sm'
                : 'bg-navy-950 text-slate-300 hover:bg-navy-800'
            }`}
          >
            All (36)
          </button>
          {[1, 2, 3, 4, 5].map((mod) => (
            <button
              key={mod}
              onClick={() => setSelectedModule(mod)}
              className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all ${
                selectedModule === mod
                  ? 'bg-gold-500 text-navy-950 shadow-sm'
                  : 'bg-navy-950 text-slate-300 hover:bg-navy-800'
              }`}
            >
              M{mod} ({schedule.filter(s => s.module === mod).length}h)
            </button>
          ))}
        </div>

        {/* Status Filter & Actions */}
        <div className="flex items-center gap-3">
          <div className="flex items-center bg-navy-950 rounded-lg p-0.5 border border-slate-700">
            <button
              onClick={() => setStatusFilter('all')}
              className={`px-2.5 py-1 text-xs rounded-md font-medium ${
                statusFilter === 'all' ? 'bg-burgundy-800 text-gold-200' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              All
            </button>
            <button
              onClick={() => setStatusFilter('completed')}
              className={`px-2.5 py-1 text-xs rounded-md font-medium ${
                statusFilter === 'completed' ? 'bg-emerald-900 text-emerald-200' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Completed
            </button>
            <button
              onClick={() => setStatusFilter('pending')}
              className={`px-2.5 py-1 text-xs rounded-md font-medium ${
                statusFilter === 'pending' ? 'bg-amber-900 text-amber-200' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Pending
            </button>
          </div>

          <button
            onClick={exportCSV}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-navy-950 hover:bg-slate-800 border border-slate-700 rounded-lg text-xs font-semibold text-slate-200 transition-all"
          >
            <Download className="w-3.5 h-3.5 text-gold-400" />
            <span>Export CSV</span>
          </button>
        </div>
      </div>

      {/* Schedule Table */}
      <div className="rounded-xl border border-gold-500/20 bg-navy-900/60 overflow-hidden shadow-xl">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-300">
            <thead className="bg-navy-950/90 text-gold-300 uppercase tracking-wider font-semibold border-b border-gold-500/20">
              <tr>
                <th className="py-3 px-3 w-14 text-center">Day</th>
                <th className="py-3 px-3 w-28">Date &amp; Day</th>
                <th className="py-3 px-3 w-28">Period</th>
                <th className="py-3 px-3 w-20 text-center">Mod / Topic</th>
                <th className="py-3 px-4">Lecture Title &amp; Pedagogical Scope</th>
                <th className="py-3 px-3 w-16 text-center">CO</th>
                <th className="py-3 px-3 w-24 text-center">Bloom's</th>
                <th className="py-3 px-3 w-28 text-center">Status</th>
                <th className="py-3 px-3 w-24 text-center">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/80">
              {filteredSchedule.map((item) => {
                const isCompleted = completedDays.has(item.day_number);
                const isTopicDuplication = item.day_number === 14 || item.day_number === 15;

                return (
                  <tr 
                    key={item.day_number}
                    className={`hover:bg-slate-800/40 transition-colors ${
                      isCompleted ? 'bg-emerald-950/10' : ''
                    } ${isTopicDuplication ? 'border-l-4 border-amber-400' : ''}`}
                  >
                    {/* Day Number */}
                    <td className="py-3 px-3 text-center font-bold text-slate-200">
                      <span className="inline-block px-2 py-0.5 rounded bg-navy-950 border border-slate-700">
                        #{item.day_number.toString().padStart(2, '0')}
                      </span>
                    </td>

                    {/* Date & Day */}
                    <td className="py-3 px-3 whitespace-nowrap">
                      <div className="font-semibold text-slate-200">{item.date_display}</div>
                      <div className="text-[11px] text-slate-400">{item.day_name}</div>
                    </td>

                    {/* Period */}
                    <td className="py-3 px-3 whitespace-nowrap">
                      <div className="text-[11px] text-slate-300 font-medium">{item.period.split('(')[0]}</div>
                      <div className="text-[10px] text-slate-500">
                        {item.period.includes('(') ? item.period.split('(')[1].replace(')', '') : ''}
                      </div>
                    </td>

                    {/* Mod / Topic */}
                    <td className="py-3 px-3 text-center whitespace-nowrap">
                      <span className="px-2 py-0.5 rounded text-[11px] font-bold bg-burgundy-900/80 text-gold-300 border border-gold-500/20">
                        M{item.module} : {item.topic_id}
                      </span>
                    </td>

                    {/* Title */}
                    <td className="py-3 px-4">
                      <div className="font-semibold text-slate-100 hover:text-gold-300 cursor-pointer flex items-center gap-1.5"
                           onClick={() => onSelectDay(item.day_number)}>
                        <span>{item.title}</span>
                        {isTopicDuplication && (
                          <span className="px-1.5 py-0.2 rounded text-[10px] bg-amber-500/20 text-amber-300 border border-amber-500/30">
                            Flagged 2.7/2.8
                          </span>
                        )}
                      </div>
                      <div className="text-[11px] text-slate-400 mt-0.5">
                        {item.module_name} &bull; 1.0 Lecture Hour
                      </div>
                    </td>

                    {/* CO */}
                    <td className="py-3 px-3 text-center">
                      <span className="font-bold text-gold-300 bg-navy-950 px-2 py-0.5 rounded border border-slate-700">
                        {item.co}
                      </span>
                    </td>

                    {/* Bloom's */}
                    <td className="py-3 px-3 text-center">
                      <span className="text-[11px] text-slate-300 bg-slate-900 px-2 py-0.5 rounded border border-slate-700">
                        {item.blooms.split(' ')[0]}
                      </span>
                    </td>

                    {/* Status Toggle */}
                    <td className="py-3 px-3 text-center">
                      <button
                        onClick={() => onToggleStatus(item.day_number)}
                        className={`px-2.5 py-1 rounded-full text-[11px] font-semibold flex items-center gap-1.5 mx-auto transition-all ${
                          isCompleted
                            ? 'bg-emerald-950 text-emerald-300 border border-emerald-500/40 hover:bg-emerald-900'
                            : 'bg-amber-950/60 text-amber-300 border border-amber-500/30 hover:bg-amber-900/80'
                        }`}
                      >
                        {isCompleted ? (
                          <>
                            <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                            <span>Done</span>
                          </>
                        ) : (
                          <>
                            <Clock className="w-3 h-3 text-amber-400" />
                            <span>Pending</span>
                          </>
                        )}
                      </button>
                    </td>

                    {/* Action Button */}
                    <td className="py-3 px-3 text-center">
                      <button
                        onClick={() => onSelectDay(item.day_number)}
                        className="p-1.5 rounded-lg bg-navy-950 hover:bg-burgundy-900 border border-gold-500/30 text-gold-300 hover:text-gold-200 transition-all inline-flex items-center gap-1 text-[11px] font-semibold"
                        title="Open Day Lesson Package"
                      >
                        <FileText className="w-3.5 h-3.5" />
                        <span>Package</span>
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {filteredSchedule.length === 0 && (
          <div className="p-8 text-center text-slate-400">
            <p className="text-sm">No lecture sessions match the selected filters.</p>
            <p className="text-xs text-slate-500 mt-1">Try clearing your search term or changing the module filter.</p>
          </div>
        )}
      </div>

    </div>
  );
};

