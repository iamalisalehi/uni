
%%
%% This is file `iut-thesis.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% xepersian.dtx  (with options: `vahid-thesis.cls')
%% 
%%   __________________________________
%%   Copyright © 2008-2009 Vafa Khalighi
%%  
%%   Edited by Vahid Damanafshan    <vdamanafshan@yahoo.com>
%%  
%%   License information appended.
%% 
%% 
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{iut-thesis}
              [2009/02/09 v0.1 Persian thesis document class in XeLaTeX]
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}
\ProcessOptions
\LoadClass{book}
\def\university#1{\gdef\@university{#1}}
\def\department#1{\gdef\@department{#1}}
\def\degree#1{\gdef\@degree{#1}}
\def\thesisdate#1{\gdef\@thesisdate{#1}}
\def\defenddate#1{\gdef\@defenddate{#1}}
\def\supervisor#1{\gdef\@supervisor{#1}}
\def\secsupervisor#1{\gdef\@secsupervisor{#1}}
\def\supervisorM#1{\gdef\@supervisorM{#1}}
\def\secsupervisorM#1{\gdef\@secsupervisorM{#1}}
\def\advisorM#1{\gdef\@advisorM{#1}}
\def\secadvisorM#1{\gdef\@secadvisorM{#1}}
\def\advisor#1{\gdef\@advisor{#1}}
\def\secadvisor#1{\gdef\@secadvisor{#1}}
\def\refreeinA#1{\gdef\@refreeinA{#1}}
\def\refreeinB#1{\gdef\@refreeinB{#1}}
\def\refreeoutA#1{\gdef\@refreeoutA{#1}}
\def\refreeoutB#1{\gdef\@refreeoutB{#1}}
\def\tit#1{\gdef\@tit{#1}}
\def\type#1{\gdef\@type{#1}}
\def\city#1{\gdef\@city{#1}}
\def\subject#1{\gdef\@subject{#1}}
\def\field#1{\gdef\@field{#1}}
\def\department#1{\gdef\@department{#1}}
\def\DGC#1{\gdef\@DGC{#1}}
% تنظیمات و تغییر متغییرهای لاتین
\def\latintitle#1{\gdef\@latintitle{#1}}
\def\latinauthor#1{\gdef\@latinauthor{#1}}
\def\latindegree#1{\gdef\@latindegree{#1}}
\def\latinfaculty#1{\gdef\@latinfaculty{#1}}
\def\latinthesisdate#1{\gdef\@latinthesisdate{#1}}
\def\latintype#1{\gdef\@latintype{#1}}
\def\latindefenddate#1{\gdef\@latindefenddate{#1}}
\def\latinsupervisor#1{\gdef\@latinsupervisor{#1}}
\def\latinadvisorM#1{\gdef\@latinadvisorM{#1}}
\def\latinsupervisorM#1{\gdef\@latinsupervisorM{#1}}
\def\latinadvisor#1{\gdef\@latinadvisor{#1}}
\def\latinrefreeinA#1{\gdef\@latinrefreeinA{#1}}
\def\latinrefreeinB#1{\gdef\@latinrefreeinB{#1}}
\def\latindefenddate#1{\gdef\@latindefenddate{#1}}
\def\seclatinsupervisor#1{\gdef\@seclatinsupervisor{#1}}
\def\seclatinadvisorM#1{\gdef\@seclatinadvisorM{#1}}
\def\seclatinsupervisorM#1{\gdef\@seclatinsupervisorM{#1}}
\def\seclatinadvisor#1{\gdef\@seclatinadvisor{#1}}
\def\latinrefreeoutA#1{\gdef\@latinrefreeoutA{#1}}
\def\latinrefreeoutB#1{\gdef\@latinrefreeoutB{#1}}
\def\latinDGC#1{\gdef\@latinDGC{#1}}
\def\latincity#1{\gdef\@latincity{#1}}
\def\latinuniversity#1{\gdef\@latinuniversity{#1}}
\def\pagenumber#1{\gdef\@pagenumber{#1}}
\def\email#1{\gdef\@email{#1}}
\def\lang#1{\gdef\@lang{#1}}
\def\supervisoremail#1{\gdef\@supervisoremail{#1}}
%:کلمات رزرو
\def\onvan{عنوان}
\def\writtenfor{}
\def\undersupervision{استاد درس}
\def\undersupervisions{استاد درس}
\def\coadvisor{}
\def\coadvisors{}
\def\latinundersupervision{Supervisor}
\def\latinundersupervisions{Supervisors}
\def\latincoadvisor{Advisor}
\def\latincoadvisors{Advisors}
\def\by{نگارنده}
\def\latinby{by}
\def\latinwrittenfor{DISSERTATION SUBMITTED IN PARTIAL
\\[.2cm]
 FULFILLMENT OF THE
REQUIREMENTS FOR THE 
\\[.2cm]
DEGREE OF}

\def\makelatinabstract{\newpage
\thispagestyle{empty}
\vskip 15mm
\begin{center}{\Large\bfseries \@latintitle \\[5mm]}
{\large \@latinauthor}\par
{\small \@email}\par
{\@latindefenddate}\par
{\@latinfaculty}\par
{\@latinuniversity,Isfahan 84156-83111,Iran}\par
{University Code: IUT-77142}\par
\end{center}
\hspace{0.5cm}
{Degree:\@latindegree \hfill Language:\@lang }\par
{supervisor: \@latinsupervisor}\par
{\@supervisoremail \hfill }\par
% {\bfseries Abstract}
\vskip 1cm
{\bfseries\large Abstract}\par
}
%%=================================================طراحی صفحه عنوان  و جلد فارسی
\def\makefatitle{\begin{titlepage}
%\topmargin=-30pt
\vspace{-1cm}\centerline{{\includegraphics[height=3cm]{logo.pdf}}}
\centerline{\large \bfseries\@university}
\vspace{3pt}
\centerline{\large\bfseries\@department}
\vskip .9cm
{\large\bfseries\@type}
\vskip .3cm
{\large\bfseries\writtenfor
\@degree}
{\large\bfseries{}}
\vskip .2cm
\centerline{\large \bfseries\@subject%
%، گرایش
%\@field
}
\vskip 1cm
%\baselineskip=.750cm
\centerline{\Large \bfseries\onvan}


\vskip .4cm
{\Huge\titr \@title}\par
{\Huge\titr \@tit}
\vskip .8cm


{\large%
  \by\par}
\vskip 6.5mm
{\nil\Large \@author}\par
\vskip 8mm
{\large
\ifx\@secsupervisor\empty
  \undersupervision\par
   \nil \Large \@supervisor\par
  \else
  \undersupervisions\par
  \nil \Large \@supervisor\par
\vskip 5mm
 \@secsupervisor\par 
  \fi}
\vskip 8mm
{\large
\large
\ifx\@secadvisor\empty
 \coadvisor\par
   \titr \Large \@advisor\par
  \else
\coadvisors\par
  \nastaliq \Large \@advisor\par
\vskip 5mm
 \@secadvisor\par 
  \fi}

\vskip 8mm
%\par
%%\vskip 1cm
{\large \@thesisdate}
%\par
%\vskip 1cm

%\par
%{\large\bfseries\universityof\space\@university \par\@city}
%\vfill
\end{titlepage}%
}
% در این بخش صفحه دفاعیه طراحی شده است
\def\approval{
\begin{titlepage}
\begin{center}
\vspace{-1cm}\centerline{{\includegraphics[height=3cm]{logo.pdf}}}
{\Large \@university} \\
{\Large \@department}
\vskip 1cm

\large{\@type\space \@degree}
\vskip 2cm
{\Large\titr \@title}\par
{\Large\titr \@tit}
\vskip 1cm
\by:\space\@author
\end{center}
\vfill
در تاریخ 
\@defenddate
توسط کمیته‌ی تخصصی زیر مورد بررسی و تصویب نهایی قرار گرفت.
\vskip 1cm
\renewcommand{\arraystretch}{3}
\begin{tabular}{p{10cm}p{2cm}}
%اگر اساتید راهنما یا مشاور داردید باید این قسمت را خوب بررسی کنید
 \ifx\@secsupervisorM\empty
 \textbf{استاد راهنما}: \@supervisorM  & امضاء: \\
\else
 \textbf{استاد راهنما}: \@supervisorM  & امضاء: \\
\textbf{استاد راهنما}: \@secsupervisorM  & امضاء: \\
\fi
\ifx\@secadvisor\empty
\textbf{استاد مشاور}: \@advisorM  & امضاء: \\
\else
\textbf{استاد مشاور}: \@advisorM  & امضاء: \\
\textbf{استاد مشاور}: \@secadvisorM  & امضاء: \\
\fi
\textbf{استاد ممتحن مدعو}: \@refreeinA & امضاء: \\
\textbf{استاد ممتحن داخلی }: \@refreeinB & امضاء: \\
 \ifx\@refreeoutA\empty
 \empty
 \else
 \textbf{استاد ممتحن خارجی ۱}: \@refreeoutA & امضاء: \\
 \fi
  \ifx\@refreeoutB\empty
 \empty
 \else
 \textbf{استاد ممتحن خارجی۲}: \@refreeoutB & امضاء: \\
 \fi
%\textbf{استاد ممتحن خارجی ۲}: \@refreeoutB & امضاء:
\textbf{سرپرست تحصیلات تکمیلی}: \@DGC & امضاء: \\
\end{tabular}
\renewcommand{\arraystretch}{1}
\end{titlepage}
% در این بخش صفحه اظهارنامه و حقوق طراحی شده است
\chapter*{حقوق و مالکیت}
کلیه حقوق مالکیت مادی و معنوی مربوط به اين پايان‌نامه متعلق به دانشگاه صنعتی اصفهان و پدیدآورندگان است. این حقوق توسط دانشگاه صنعتی اصفهان و بر اساس خط مشی مالکیت فکری این دانشگاه، ارزش‌گذاری و سهم‌بندی خواهد شد.
هر گونه بهره‌برداری از محتوا، نتايج یا اقدام برای تجاری‌سازی دستاوردهای اين پايان‌نامه تنها با مجوز کتبی دانشگاه صنعتی اصفهان امکان‌پذیر است.

\phantom{a}  \hfill \@author ~~~ \\[4pt]
\phantom{a} \hfill امضاء~~~
}
%در این بخش صفحه مقدمه مجزا طراحی شده است.
\newcommand{\preface}{
\chapter*{مقدّمه}
%\pagenumbering{arabic}\setcounter{page}{1}
% \rhead{مقدّمه}
\addcontentsline{toc}{chapter}{مقدّمه}
\def\theequation{\arabic{equation}}}
%
\newcommand{\EndPreface}{
\def\theequation{\arabic{equation}-\thechapter}}
%در این بخش صفحه سپاسگزاری طراحی شده است.
\newcommand{\acknowledge}{
\chapter*{سپاس‌گزاری}
\rhead{سپاس‌گزاری}
\addcontentsline{toc}{chapter}{سپاس‌گزاری}}

\def\titlepage{\newpage\centering
  \thispagestyle{empty}
  \parindent 0pt \parskip 10pt plus 1fil minus 1fil
  \def\baselinestretch{1}\@normalsize\vbox to \vsize\bgroup\vbox to 9in\bgroup}
\def\endtitlepage{\par\kern 0pt\egroup\vss\egroup\newpage}

\def\signature#1#2{\par\noindent#1\dotfill\null\\*
  {\raggedright #2\par}}

\def\abstract{\subsection*{\abstractname}\small
\@normalsize}
\def\endabstract{\par}

\pagenumbering{adadi}

\def\keywords#1{\par\vspace{5mm}
\noindent {\textbf{واژه‌های کلیدی:}} {\textit{ #1}}}

\def\latinkeywords#1{\par \vspace{5mm}
\noindent {\textbf{Keywords:}} {\textit{ #1}}}

\font\titlefont=cmssbx10 scaled 2074
\font\supervisorfont=cmbxti10 
%==========================================english defend page
\def\makelatindeftitle{
\begin{latin}
\begin{titlepage}
{\includegraphics[height=4cm]{arm_en.eps}}
{\centerline{\large\bfseries\@latinuniversity} 
%\vskip 1.5cm
%\addtolength{\baselineskip}{5mm}
\@latinfaculty\space 
\vskip 5mm
%\vskip 1cm
{\titlefont \@latintitle} \par
\addtolength{\baselineskip}{-5mm}
\vskip 7mm
\centering{A Thesis 

Submitted in partial fulfillment of the requirements 

for the degree of Master of Science}
\vskip 7mm
{\bfseries\latinby}\par

{\Large \supervisorfont\@latinauthor}\par
\vskip 1.5cm
%\vskip 1m
\vfill
\par}
{Evaluated and Approved by the Thesis Committee, on \@latindefenddate}
\vskip 1.cm
\centering{
\renewcommand{\arraystretch}{2}
\begin{tabular}{p{13cm}p{2cm}}
%اگر اساتید راهنما یا مشاور داردید باید این قسمت را خوب بررسی کنید
{Supervisor}: \@latinsupervisorM  & sign \\
\ifx \@seclatinsupervisorM\empty
\empty
\else
{Supervisor}: \@seclatinsupervisorM  & sign \\
\fi
{Advisor}: \@latinadvisorM  & sign \\
\ifx \@seclatinadvisorM\empty
\empty
\else
{Advisor}: \@seclatinadvisorM  & sign \\
\fi
{Examiner}: \@latinrefreeinA & sign \\
{Examiner}: \@latinrefreeinB & sign \\
\if \@latinrefreeoutA\empty
\empty
\else
{Examiner}: \@latinrefreeoutA  & sign \\
\fi
\if \@latinrefreeoutB\empty
\empty
\else
{Examiner}: \@latinrefreeoutB  & sign \\
\fi
{Department Graduate Coordinateor}: \@latinDGC & sign \\
\end{tabular}
\renewcommand{\arraystretch}{1}
}
\end{titlepage}%
\end{latin}
}
%===========================================latin back page
\def\makelatintitle{\begin{titlepage}
{\includegraphics[height=4cm]{arm_en.eps}}
\centerline{\large\bfseries\@latinuniversity} 
\@latinfaculty\space 
\large\bfseries
\vskip 5mm
{\latinwrittenfor\par
\large\@latindegree}
\vskip 1cm
{\titlefont \@latintitle} \par
\addtolength{\baselineskip}{-5mm}
\vskip 7mm

{\ifx\@seclatinsupervisor\empty
\latinundersupervision \par
\supervisorfont\@latinsupervisor
\else
\latinundersupervisions \par
\supervisorfont\@latinsupervisor\par
\supervisorfont\@seclatinsupervisor
\fi}
\vskip 4mm
{\ifx\@secadvisor\empty
\latincoadvisor\par
\supervisorfont\@latinadvisor\par
\else
\latincoadvisors\par
\supervisorfont\@latinadvisor\par
\supervisorfont\@seclatinadvisor
\fi}
\vskip 5mm
{\bfseries\latinby}\par
{\Large \supervisorfont\@latinauthor}\par
\vskip 5mm

\par
\vskip 1cm

\par
\vskip 1cm
\@latincity

{\@latinthesisdate}
\par
\vskip 1cm

\par


\vfill
\end{titlepage}%
}


%\def\endlatinabstract{\newpage}
%% 
%% Copyright © 2008-2009      by Vafa Khalighi     <vafa@users.berlios.de>
%% 
%% Distributable under the LaTeX Project Public License,
%% version 1.3c or higher (your choice). The latest version of
%% this license is at: http://www.latex-project.org/lppl.txt
%% 
%% This work is "maintained" (as per LPPL maintenance status)
%% by Vafa Khalighi.
%% 
%% 
%% 
%% 
%%
%% End of file `xepersian-thesis.cls'.
