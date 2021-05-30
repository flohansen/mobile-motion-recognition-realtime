\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{hsflensburg}[2020/04/13 Flensburg Document Class]

\LoadClass[12pt,a4paper]{scrreprt}

\RequirePackage{graphicx}
\RequirePackage{setspace}

\newcommand{\name}[1]{{\normalsize{#1}}}
\newcommand{\institution}[1]{{\footnotesize{#1}}}

\makeatletter
\renewcommand{\maketitle}{
	\begin{titlepage}
		\centering
		\IfFileExists
			{./images/logo.png}
			{\includegraphics[width=0.5\textwidth]{logo}\par}
			{\vfill}
		\vspace{3cm}
		{\large\bfseries \@title\par}
		\vspace{1cm}
		{\huge\bfseries \@subtitle\par}
		\vspace{3cm}

		{\normalsize \lineskip .75em
		\begin{tabular}[t]{c}
			\@author
		\end{tabular}\par}

		\vfill
		\today
	\end{titlepage}
}
\makeatother

\setlength{\parindent}{0cm}
\setlength{\parskip}{0.15cm}
\graphicspath{{images/}}
\onehalfspacing
