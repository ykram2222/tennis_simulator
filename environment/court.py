import numpy as np

def court_segments(width=10.97, length=23.77):
    x0, x1 = -width/2, width/2
    y0, y1 = 0.0, length
    sx0, sx1 = -8.23/2, 8.23/2
    sy0, sy1 = length/2 - 6.40, length/2 + 6.40
    pairs = [
        ((x0,y0,0),(x1,y0,0)), ((x0,y1,0),(x1,y1,0)),
        ((x0,y0,0),(x0,y1,0)), ((x1,y0,0),(x1,y1,0)),
        ((sx0,y0,0),(sx0,y1,0)), ((sx1,y0,0),(sx1,y1,0)),
        ((sx0,sy0,0),(sx1,sy0,0)), ((sx0,sy1,0),(sx1,sy1,0)),
        ((x0,length/2,0),(x1,length/2,0))]
    return [(np.array(a,float), np.array(b,float)) for a,b in pairs]
