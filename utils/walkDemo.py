# -*- coding: utf-8 -*-
"""
Demonstrates a variety of uses for ROI. This class provides a user-adjustable
region of interest marker. It is possible to customize the layout and 
function of the scale/rotate handles in very flexible ways. 
"""

# import initExample ## Add path to library (just for examples; you do not need this)

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import math
import sys

from pyqtgraph.graphicsItems.ScatterPlotItem import Symbols

sys.path.insert(0,'')

from utils.experiment import Experiment

pg.setConfigOptions(antialias=True)

def getPositionOnCircle(x0,y0,r,theta):
    theta = 270-theta
    x = x0 + r * math.cos(theta * math.pi/180)
    y = y0 + r * math.sin(theta * math.pi/180)
    return x, y

pg.setConfigOptions(imageAxisOrder='row-major')

exp = Experiment('../data/raw_txt/SL02-T03')
jointdataGHL = exp.jointsData.GHL.values
jointdataGKL = exp.jointsData.GKL.values
jointdataGAL = exp.jointsData.GAL.values

jointdataGHR = exp.jointsData.GHR.values
jointdataGKR = exp.jointsData.GKR.values
jointdataGAR = exp.jointsData.GAR.values

jointdataPHL = exp.jointsData.PHL.values
jointdataPKL = exp.jointsData.PKL.values
jointdataPAL = exp.jointsData.PAL.values

jointdataPHR = exp.jointsData.PHR.values
jointdataPKR = exp.jointsData.PKR.values
jointdataPAR = exp.jointsData.PAR.values

time = exp.jointsData.Time.values
## create GUI
app = QtGui.QApplication([])
w = pg.GraphicsLayoutWidget(show=True, size=(1000,800), border=True)


w.setWindowTitle('MoBIShow-Walking')
    
w2 = w.addLayout(row=0, col=1)
v2a = w2.addViewBox(row=1, col=0, lockAspect=True)
plotGL = pg.PlotDataItem()
plotGR= pg.PlotDataItem()
plotGL.setData(np.asarray([0,10,10,10]), np.asarray([0,10,10, -20]), pen=pg.mkPen(color='w',width=1), symbol='s', symbolSize=8, symbolPen=pg.mkPen(color='#0ffef9'), symbolBrush=pg.mkBrush(color='#b7fffa00'))
plotGR.setData(np.asarray([0,10,10,10]), np.asarray([0,10,10, -20]), pen=pg.mkPen(color='w',width=1), symbol='o', symbolSize=8, symbolPen=pg.mkPen(color='#0ffef9'), symbolBrush=pg.mkBrush(color='#b7fffa00'))

plotPL = pg.PlotDataItem()
plotPR= pg.PlotDataItem()
plotPL.setData(np.asarray([0,10,10,10]), np.asarray([0,10,10, -20]), pen=pg.mkPen(color='w',width=1), symbol='s', symbolSize=8, symbolPen=pg.mkPen(color='#fc86aa'), symbolBrush=pg.mkBrush(color='#b7fffa00'))
plotPR.setData(np.asarray([0,10,10,10]), np.asarray([0,10,10, -20]), pen=pg.mkPen(color='w',width=1), symbol='o', symbolSize=8, symbolPen=pg.mkPen(color='#fc86aa'), symbolBrush=pg.mkBrush(color='#b7fffa00'))

v2a.addItem(plotGL)
v2a.addItem(plotGR)
v2a.addItem(plotPL)
v2a.addItem(plotPR)

start = -7
margin = 6
legendGL= pg.PlotDataItem()
legendGL.setData(np.asarray([start+margin]), np.asarray([3]), pen=pg.mkPen(color='w',width=1), symbol='s', symbolSize=8, symbolPen=pg.mkPen(color='#0ffef9'), symbolBrush=pg.mkBrush(color='#b7fffa00'))

legendGR= pg.PlotDataItem()
legendGR.setData(np.asarray([start+margin*2]), np.asarray([3]), pen=pg.mkPen(color='w',width=1), symbol='o', symbolSize=8, symbolPen=pg.mkPen(color='#0ffef9'), symbolBrush=pg.mkBrush(color='#b7fffa00'))

legendPL= pg.PlotDataItem()
legendPL.setData(np.asarray([start+margin*3]), np.asarray([3]), pen=pg.mkPen(color='w',width=1), symbol='s', symbolSize=8, symbolPen=pg.mkPen(color='#fc86aa'), symbolBrush=pg.mkBrush(color='#b7fffa00'))

legendPR= pg.PlotDataItem()
legendPR.setData(np.asarray([start+margin*4]), np.asarray([3]), pen=pg.mkPen(color='w',width=1), symbol='o', symbolSize=8, symbolPen=pg.mkPen(color='#fc86aa'), symbolBrush=pg.mkBrush(color='#b7fffa00'))

v2a.addItem(legendGL)
v2a.addItem(legendGR)
v2a.addItem(legendPL)
v2a.addItem(legendPR)

space = 0.1
textGL = pg.TextItem('Actual Left')
textGL.setPos(start+margin+space, 3.5)
v2a.addItem(textGL)

textGR = pg.TextItem('Actual Right')
textGR.setPos(start+margin*2+space, 3.5)
v2a.addItem(textGR)

textPL = pg.TextItem('Predicted Left')
textPL.setPos(start+margin*3+space, 3.5)
v2a.addItem(textPL)

textPR = pg.TextItem('Predicted Right')
textPR.setPos(start+margin*4+space, 3.5)
v2a.addItem(textPR)

v2a.addItem(textGL)
v2a.addItem(textGR)
v2a.addItem(textPL)
v2a.addItem(textPR)

# v2a.addItem(cir)
v2a.disableAutoRange('xy')
v2a.autoRange()

a = list(range(90))
i = 7000
def update():
    global i, plotGL, plotGR, plotPL, plotPR
    v2a.removeItem(plotGL)
    v2a.removeItem(plotGR)
    v2a.removeItem(plotPL)
    v2a.removeItem(plotPR)

    GHxL,GHyL = getPositionOnCircle(0,0,4,jointdataGHL[i])
    GKxL,GKyL = getPositionOnCircle(GHxL,GHyL, 5,jointdataGKL[i])
    GKxLShort,GKyLShort = getPositionOnCircle(GHxL,GHyL, 4.5,jointdataGKL[i])
    GAx1L,GAy1L = getPositionOnCircle(GKxL,GKyL, 1, 270+jointdataGAL[i])
    GAx2L,GAy2L = getPositionOnCircle(GKxL,GKyL, 0.5, 270+jointdataGAL[i]+180)
    plotGL.setData(np.asarray([0,GHxL,GKxLShort,GAx1L,GAx2L,GKxLShort]), np.asarray([0,GHyL,GKyLShort,GAy1L,GAy2L,GKyLShort]))

    GHxR,GHyR = getPositionOnCircle(0,0,4,-jointdataGHR[i])
    GKxR,GKyR = getPositionOnCircle(GHxR,GHyR, 5,-jointdataGKR[i])
    GKxRShort,GKyRShort = getPositionOnCircle(GHxR,GHyR, 4.5,-jointdataGKR[i])
    GAx1R,GAy1R = getPositionOnCircle(GKxR,GKyR, 1, 270-jointdataGAR[i])
    GAx2R,GAy2R = getPositionOnCircle(GKxR,GKyR, 0.5, 270-jointdataGAR[i]+180)
    plotGR.setData(np.asarray([0,GHxR,GKxRShort,GAx1R,GAx2R,GKxRShort]), np.asarray([0,GHyR,GKyRShort,GAy1R,GAy2R,GKyRShort]))

    # P
    PHxL,PHyL = getPositionOnCircle(0,0,4,jointdataPHL[i])
    PKxL,PKyL = getPositionOnCircle(PHxL,PHyL, 5,jointdataPKL[i])
    PKxLShort,PKyLShort = getPositionOnCircle(PHxL,PHyL, 4.5,jointdataPKL[i])
    PAx1L,PAy1L = getPositionOnCircle(PKxL,PKyL, 1, 270+jointdataPAL[i])
    PAx2L,PAy2L = getPositionOnCircle(PKxL,PKyL, 0.5, 270+jointdataPAL[i]+180)
    plotPL.setData(np.asarray([0,PHxL,PKxLShort,PAx1L,PAx2L,PKxLShort])+20, np.asarray([0,PHyL,PKyLShort,PAy1L,PAy2L,PKyLShort]))

    PHxR,PHyR = getPositionOnCircle(0,0,4,-jointdataPHR[i])
    PKxR,PKyR = getPositionOnCircle(PHxR,PHyR, 5,-jointdataPKR[i])
    PKxRShort,PKyRShort = getPositionOnCircle(PHxR,PHyR, 4.5,-jointdataPKR[i])
    PAx1R,PAy1R = getPositionOnCircle(PKxR,PKyR, 1, 270-jointdataPAR[i])
    PAx2R,PAy2R = getPositionOnCircle(PKxR,PKyR, 0.5, 270-jointdataPAR[i]+180)
    plotPR.setData(np.asarray([0,PHxR,PKxRShort,PAx1R,PAx2R,PKxRShort])+20, np.asarray([0,PHyR,PKyRShort,PAy1R,PAy2R,PKyRShort]))

    v2a.addItem(plotGL)
    v2a.addItem(plotGR)
    v2a.addItem(plotPL)
    v2a.addItem(plotPR)
    i+=2
    # print(time[i])
        

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(10)

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
