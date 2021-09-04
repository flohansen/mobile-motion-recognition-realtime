\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{hsflensburg}[2020/04/13 Flensburg Document Class]

\LoadClass[12pt,a4paper]{scrreprt}

\RequirePackage{graphicx}
\RequirePackage{setspace}
\RequirePackage{tabularx}

\newcommand{\name}[1]{{\normalsize{#1}}}
\newcommand{\institution}[1]{{\footnotesize{#1}}}

\makeatletter
\renewcommand{\maketitle}{
	\begin{titlepage}
		\centering
		{\large\bfseries Hochschule Flensburg\par}
		\vspace{0.6cm}
		{\huge\bfseries \@subtitle\par}
		\vspace{3cm}

		\begin{tabular}{p{1.5cm}p{13cm}}
			Thema: & \@title \\[3cm]
			von: & Florian Hansen
		\end{tabular}

		\vspace{3cm}

		\begin{tabular}{b{3cm}b{11.5cm}}
			Matrikel-Nr.: & 680169 \\[0.1cm] 
			Studiengang: & Angewandte Informatik \\[0.1cm]
			Betreuer/in und Erstbewerter/in: & Prof. Dr. rer. nat. Tim Aschmoneit \\[0.1cm]
			Zweitbewerter/in: & Prof. Dr. rer. nat. Torben Wallbaum \\[0.1cm]
			Ausgabedatum: & 06.04.2021 \\[0.1cm]
			Abgabedatum: & 06.09.2021 \\[0.1cm]
		\end{tabular}

		\vfill
	\end{titlepage}
}
\makeatother

\setlength{\parindent}{0cm}
\setlength{\parskip}{0.15cm}
\graphicspath{{images/}}
\onehalfspacing
