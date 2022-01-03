# coating
Study effect of coating

Assuming you have three file MEA1.h5, MEA2.h5 and MEA3.h5, you can launch Spyking circus on all your files like that

> spyking-circus MEA1.h5
> spyking-circus MEA1.h5 -m thresholing

You need to edit the .params first to be sure that [no_edits] filter_done is set to False, if you start with unfiltered data for the first time. 

Once the data have been generated, the script.py will make the plots
