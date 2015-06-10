*Master's Course, SS2015
Faculty of Physics and Astronomy, University of Heidelberg*

# Computational Statistics and Data Analysis (MVComp2)

Course LSF entry

Lecturer: PD Dr. Coryn Bailer-Jones

Assistants: Dr. Morgan Fouesneau, Dr. Dae-Won Kim

Summer semester 2015

This page provides the homework assignments in a form of python **notebooks**.
You can click in the table below to read it online or download them. (python is
not imposed to solve the exercises.)

This course and exercises take a pragmatic approach to using statistics and
computational methods to analyse data. The focus will be on concepts,
understanding problems, and the application of techniques to solving problems,
rather than reproducing proofs or teaching you recipes to memorize.

The course website is available
[here](http://www.mpia.de/homes/calj/compstat_ss2015/main.html)

**This repository gives the homeworks related datasets and eventually
corrections** (table below for links)

The repository will be updated **after each class** to give the assignments. All
datasets, gists of code will also be included.  Examples of solutions (hardly
unique) will be included eventually.

## Some homework guidelines

**Notebooks have no meaning of imposing a format** to give us back your
homework assignments. Instead they give me convenient ways to keep both texts
and codes at the same place.

* Each week, we will mark your homework on a scale of 100 points in total.
  (details given with the exercises)

* You are allowed to work in groups of at most 3 persons and return 1 document
  per group.

* Homework documents must be returned each Tuesday. 

* **We do not mark your coding skills**.

* This means **we do not read the codes**. We do not look out for comments in
  the codes, but **we will not guess** what a plot means. Be explicit and
  describe even in once sentence what you did.

* Feel free to use the notebooks (it may not be the most efficient), be careful
  when printing (Check out `nbconvert` to produce a pdf or even latex document).

## Computing language

* **We do not impose a language**. Feel free to use any that you judge efficient
  for you.  Obviously we cannot provide full support, nor we cannot give full
  tutorials.

* If you use R, many examples of code will be included in the lecture notes. If
  you use Python, all the exercises will be using python (when coding is
  required). 

* examples in R from the course are available here: [link](http://www.mpia.de/homes/calj/compstat_ss2015/Rcodes.zip) (will be updated throughout the course)

### Online tools

In case you cannot/do not want to install libraries or softwares on your
computer, some **free** online services exist, such as:

[Sage Cloud](https://cloud.sagemath.com): python, R, and other languages

[Wakari](https://wakari.io/) Python only.


### MCMC libraries

some libraries that you may find useful later depending on your language.

[emcee (python)](http://dan.iel.fm/emcee/current/) 

[STAN (C/C++)](http://mc-stan.org/)

## Lectures

There will be 12 lectures on the following dates (the exercise session is on the
following day). The topics allocated to the dates may well change!


As github now integrates `nbviewer` **If a notebook is not accessible through
the links in the table, you can instead click on the files**

| Lecture date    | Topic                                                     | Exercises                                                                                    | datasets & snippets                                                                        | 
| --------------- | --------------------------------------------------------- | -------------------------------------------------------------------------------------------- | --------------------                                                                       | 
| 14 April        | Introduction and probability basics                       | [notebook](http://nbviewer.ipython.org/github/mfouesneau/mvcomp2/blob/master/chap1_ex.ipynb) | [rvs.dat](https://raw.githubusercontent.com/mfouesneau/mvcomp2/master/rvs.dat)             | 
| 21 April        | Estimation and error: describing data and distributions   | [notebook](http://nbviewer.ipython.org/github/mfouesneau/mvcomp2/blob/master/chap2_ex.ipynb) | [star.csv](https://raw.githubusercontent.com/mfouesneau/mvcomp2/master/star.csv)           | 
| 28 April        | Statistical models and inference                          | [notebook](http://nbviewer.ipython.org/github/mfouesneau/mvcomp2/blob/master/chap3_ex.ipynb) | [hipparcos.dat](https://raw.githubusercontent.com/mfouesneau/mvcomp2/master/hipparcos.dat) | 
| 5 May           | Linear models and regression                              | [notebook](http://nbviewer.ipython.org/github/mfouesneau/mvcomp2/blob/master/chap4_ex.ipynb) | [rmr_ISwR.dat](https://raw.githubusercontent.com/mfouesneau/mvcomp2/master/rmr_ISwR.dat) [sdss_sspp_sub.csv](https://raw.githubusercontent.com/mfouesneau/mvcomp2/master/sdss_sspp_sub.csv)  | 
| 12 May          | (Bayesian) Model fitting I                                | [notebook](http://nbviewer.ipython.org/github/mfouesneau/mvcomp2/blob/master/chap5_ex.ipynb) | [lighthouse.dat](https://raw.githubusercontent.com/mfouesneau/mvcomp2/master/lighthouse.dat) | 
| 19 May          | (Bayesian) Model fitting II                               | [notebook](http://nbviewer.ipython.org/github/mfouesneau/mvcomp2/blob/master/chap6_ex.ipynb) | [coinflip.dat](https://raw.githubusercontent.com/mfouesneau/mvcomp2/master/coinflip.dat)   | 
| 26 May          | MCMC                                                      | [notebook](http://nbviewer.ipython.org/github/mfouesneau/mvcomp2/blob/master/chap7_ex.ipynb) | [2Dline.dat](https://raw.githubusercontent.com/mfouesneau/mvcomp2/master/2Dline.dat) [metropolis.py](https://raw.githubusercontent.com/mfouesneau/mvcomp2/master/metropolis.py)              | 
| 2 June          | No lecture                                                |                                                                                              |                                                                                            | 
| 9 June          | Hypothesis testing                                        | [notebook](http://nbviewer.ipython.org/github/mfouesneau/mvcomp2/blob/master/chap8_ex.ipynb) | [iswr_vitcap.dat](https://raw.githubusercontent.com/mfouesneau/mvcomp2/master/iswr_vitcap.dat)       |                                                                                            | 
| 16 June         | Model Complexity                                          |                                                                                              |                                                                                            | 
| 23 June         | Nonparametric methods                                     |                                                                                              |                                                                                            | 
| 30 June         | Something else (details TBD)                              |                                                                                              |                                                                                            | 
| 7 July          | Gaussian processes                                        |                                                                                              |                                                                                            | 
| 14 July         | Study week                                                |                                                                                              |                                                                                            | 
| 21 July         | Exam (maybe; date to be decided with the participants)    |                                                                                              |                                                                                            | 
