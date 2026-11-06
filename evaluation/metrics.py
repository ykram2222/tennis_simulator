import numpy as np

def summarize_errors(errors):
    e = np.asarray(errors,float)
    if not len(e): return {"count":0}
    return {"count":int(len(e)),"mean_mm":float(np.mean(e)*1000),
            "median_mm":float(np.median(e)*1000),"p95_mm":float(np.percentile(e,95)*1000),
            "max_mm":float(np.max(e)*1000),"within_5mm_percent":float(np.mean(e<=.005)*100)}

def line_call(xy,width,length):
    x,y = xy
    return "IN" if -width/2<=x<=width/2 and 0<=y<=length else "OUT"
