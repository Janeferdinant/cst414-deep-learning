import React from 'react';
import { ShieldCheck, BookOpen, GraduationCap, Calendar, ExternalLink, Heart } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="mt-16 bg-navy-950 border-t border-gold-500/20 text-slate-400 py-10 px-4 lg:px-8 relative z-20">
      <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
        
        {/* Col 1: Course Meta */}
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <span className="px-2 py-0.5 rounded text-[11px] font-bold uppercase bg-burgundy-900 text-gold-300 border border-gold-500/30">
              KTU CST414
            </span>
            <span className="text-xs text-slate-300 font-medium">Scheme 2019 | S7 CSE</span>
          </div>
          <h2 className="text-sm font-bold text-slate-200">
            Deep Learning Academic Course System
          </h2>
          <p className="text-xs text-slate-400 leading-relaxed">
            Autonomous course planning, pedagogical day packages, and interactive visual lab built in strict accordance with the APJ Abdul Kalam Technological University curriculum.
          </p>
          <div className="pt-2 text-xs font-semibold text-gold-400/90 flex items-center gap-1.5">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            <span>OBE & Bloom's Taxonomy Compliant</span>
          </div>
        </div>

        {/* Col 2: Modules Overview */}
        <div>
          <h3 className="text-xs font-bold uppercase tracking-wider text-gold-300 mb-3 flex items-center gap-2">
            <BookOpen className="w-3.5 h-3.5" /> Course Modules (36 Hrs)
          </h3>
          <ul className="text-xs space-y-1.5 text-slate-300">
            <li><span className="text-gold-400 font-semibold">M1:</span> Neural Networks (7 hrs)</li>
            <li><span className="text-gold-400 font-semibold">M2:</span> Deep Learning (9 hrs)</li>
            <li><span className="text-gold-400 font-semibold">M3:</span> Convolutional Networks (6 hrs)</li>
            <li><span className="text-gold-400 font-semibold">M4:</span> Recurrent Networks (5 hrs)</li>
            <li><span className="text-gold-400 font-semibold">M5:</span> Application Areas (9 hrs)</li>
          </ul>
        </div>

        {/* Col 3: Standard Textbooks & References */}
        <div>
          <h3 className="text-xs font-bold uppercase tracking-wider text-gold-300 mb-3 flex items-center gap-2">
            <GraduationCap className="w-3.5 h-3.5" /> Approved References
          </h3>
          <ul className="text-xs space-y-2 text-slate-300">
            <li className="leading-snug">
              <strong className="text-slate-200">Goodfellow, Bengio & Courville:</strong> <em className="text-slate-400">Deep Learning</em>, MIT Press.
            </li>
            <li className="leading-snug">
              <strong className="text-slate-200">Charu C. Aggarwal:</strong> <em className="text-slate-400">Neural Networks and Deep Learning</em>, Springer.
            </li>
            <li className="leading-snug">
              <strong className="text-slate-200">Nikhil Buduma & N. Locascio:</strong> <em className="text-slate-400">Fundamentals of Deep Learning</em>, O'Reilly.
            </li>
          </ul>
        </div>

        {/* Col 4: Academic Disclaimer */}
        <div>
          <h3 className="text-xs font-bold uppercase tracking-wider text-amber-400 mb-3 flex items-center gap-2">
            <Calendar className="w-3.5 h-3.5" /> Provisional Schedule Notice
          </h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            The lecture timetable (18-Aug-2025 to 26-Nov-2025, Mon P2 / Wed P4 / Fri P1) is a <strong>provisional plan</strong>. Upon receipt of the approved institutional college calendar and departmental master timetable, dates can be synchronized instantly via the schedule manager.
          </p>
          <div className="mt-3 p-2 bg-navy-900 border border-slate-700/60 rounded text-[11px] text-slate-300">
            <span className="text-gold-300 font-semibold">Duplicate Topic Note:</span> Syllabus 2.7 &amp; 2.8 flagged for institutional faculty review.
          </div>
        </div>
      </div>

      {/* Mandatory Faculty Identification & Copyright Bar */}
      <div className="max-w-7xl mx-auto pt-6 border-t border-slate-800 flex flex-col md:flex-row items-center justify-between gap-4 text-xs">
        <div className="text-slate-300 text-center md:text-left">
          <p className="text-sm font-serif font-bold text-gold-200 tracking-wide">
            Jane Rose | Department of Computer Science &amp; Engineering
          </p>
          <p className="text-[11px] text-slate-400">
            APJ Abdul Kalam Technological University &bull; CST414 Deep Learning Course Portfolio
          </p>
        </div>

        <div className="text-center md:text-right text-slate-400 text-[11px]">
          <p>
            Generated with complete 324-file pedagogical assets &bull; Schemes 2019
          </p>
          <p className="text-slate-500 mt-0.5">
            Faculty Teaching Plan &bull; Student Learning Resource System &bull; All Rights Reserved
          </p>
        </div>
      </div>
    </footer>
  );
};

