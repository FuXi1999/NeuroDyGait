import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import math
import sys
import os

sys.path.insert(0,'')

demo_data_path = '../data/demo'
subject = 'SL05-T01'
method = 'tcn'
time_step = 200
demo_name = subject+'-'+method+'-TS-'+str(time_step)+'.npy'
data = np.load(os.path.join(demo_data_path, demo_name))
gt_joints = data[:,:6]
pre_joints = data[:,6:]

pg.setConfigOptions(antialias=True)

def getPositionOnCircle(x0,y0,r,theta):
    theta = 270-theta
    x = x0 + r * math.cos(theta * math.pi/180)
    y = y0 + r * math.sin(theta * math.pi/180)
    return x, y

pg.setConfigOptions(imageAxisOrder='row-major')


# load joints data
gtHLdata = gt_joints[:, 3]
gtKLdata = gt_joints[:, 4]
gtALdata = gt_joints[:, 5]
gtHRdata = gt_joints[:, 0]
gtKRdata = gt_joints[:, 1]
gtARdata = gt_joints[:, 2]

# load joints data
preHLdata = pre_joints[:, 3]
preKLdata = pre_joints[:, 4]
preALdata = pre_joints[:, 5]
preHRdata = pre_joints[:, 0]
preKRdata = pre_joints[:, 1]
preARdata = pre_joints[:, 2]

## create GUI
app = QtGui.QApplication([])
w = pg.GraphicsLayoutWidget(show=True, size=(1500,800), border=True)

w.setWindowTitle('MoBIShow-Walking')

# ----------------- curves window ----------------------------
layout_w = w.addLayout(row=0, col=3)
curve_w = layout_w.addPlot(row=0, col=0, colspan=2)
curve_w.hideAxis('left')
curve_w.hideAxis('bottom')
curve_w.setYRange(-200, 200)

# left 3 curves ground truth
curveGtHL = curve_w.plot(pen=pg.mkPen(pg.mkColor('#5566AA'), width=2))
gtHLsp  = pg.ScatterPlotItem(size=10, pxMode=True)
curve_w.addItem(gtHLsp )
curveGtKL = curve_w.plot(pen=pg.mkPen(pg.mkColor('#8DC43C'), width=2))
gtKLsp = pg.ScatterPlotItem(size=10, pen=pg.mkPen('w'), pxMode=True)
curve_w.addItem(gtKLsp)
curveGtAL = curve_w.plot(pen=pg.mkPen(pg.mkColor('#CC3333'), width=2))
gtALsp  = pg.ScatterPlotItem(size=10, pen=pg.mkPen('w'), pxMode=True)
curve_w.addItem(gtALsp )
# right 3 curves ground truth
curveGtHR = curve_w.plot(pen=pg.mkPen(pg.mkColor('#5566AA'), width=2))
gtHRsp = pg.ScatterPlotItem(size=10, pxMode=True)
curve_w.addItem(gtHRsp)
curveGtKR = curve_w.plot(pen=pg.mkPen(pg.mkColor('#8DC43C'), width=2))
gtKRsp = pg.ScatterPlotItem(size=10, pen=pg.mkPen('w'), pxMode=True)
curve_w.addItem(gtKRsp)
curveGtAR = curve_w.plot( pen=pg.mkPen(pg.mkColor('#CC3333'), width=2))
gtARsp = pg.ScatterPlotItem(size=10, pen=pg.mkPen('w'), pxMode=True)
curve_w.addItem(gtARsp)

# left 3 curves predict
curvePreHL = curve_w.plot(pen=pg.mkPen(pg.mkColor('#5566AA88'), width=2))
preHLsp  = pg.ScatterPlotItem(size=10, pxMode=True)
curve_w.addItem(preHLsp )
curvePreKL = curve_w.plot(pen=pg.mkPen(pg.mkColor('#8DC43C88'), width=2))
preKLsp = pg.ScatterPlotItem(size=10, pen=pg.mkPen('w'), pxMode=True)
curve_w.addItem(preKLsp)
curvePreAL = curve_w.plot(pen=pg.mkPen(pg.mkColor('#CC333388'), width=2))
preALsp  = pg.ScatterPlotItem(size=10, pen=pg.mkPen('w'), pxMode=True)
curve_w.addItem(preALsp )
# right 3 curves predict
curvePreHR = curve_w.plot(pen=pg.mkPen(pg.mkColor('#5566AA88'), width=2))
preHRsp = pg.ScatterPlotItem(size=10, pxMode=True)
curve_w.addItem(preHRsp)
curvePreKR = curve_w.plot(pen=pg.mkPen(pg.mkColor('#8DC43C88'), width=2))
preKRsp = pg.ScatterPlotItem(size=10, pen=pg.mkPen('w'), pxMode=True)
curve_w.addItem(preKRsp)
curvePreAR = curve_w.plot( pen=pg.mkPen(pg.mkColor('#CC333388'), width=2))
preARsp = pg.ScatterPlotItem(size=10, pen=pg.mkPen('w'), pxMode=True)
curve_w.addItem(preARsp)

# curve connection
connectHLCurve = pg.PlotDataItem(width=5)
connectKLCurve = pg.PlotDataItem(width=5)
connectALCurve = pg.PlotDataItem(width=5)
connectHRCurve = pg.PlotDataItem(width=5)
connectKRCurve = pg.PlotDataItem(width=5)
connectARCurve = pg.PlotDataItem(width=5)

curve_w.addItem(connectALCurve)
curve_w.addItem(connectHLCurve)
curve_w.addItem(connectHLCurve)
curve_w.addItem(connectARCurve)
curve_w.addItem(connectHRCurve)
curve_w.addItem(connectHRCurve)

# ----------------- walk window----------------------
walk_w = layout_w.addViewBox(row=0, col=2, lockAspect=True)
plotGL = pg.PlotDataItem(width=5)
plotGR= pg.PlotDataItem(width=5)
# connect line between gt and pre
connectHL = pg.PlotDataItem(width=5)
connectKL = pg.PlotDataItem(width=5)
connectAL = pg.PlotDataItem(width=5)
connectHR = pg.PlotDataItem(width=5)
connectKR = pg.PlotDataItem(width=5)
connectAR = pg.PlotDataItem(width=5)

plotGL.setData(np.asarray([-3,3,-3,3]), np.asarray([-11,0,0, -11]))
walk_w.addItem(plotGL)
walk_w.addItem(plotGR)
walk_w.addItem(connectAL)
walk_w.addItem(connectHL)
walk_w.addItem(connectHL)
walk_w.addItem(connectAR)
walk_w.addItem(connectHR)
walk_w.addItem(connectHR)


# legend
textHL = pg.TextItem('Hip Left')
textHL.setPos(-2.9, -9.8)
walk_w.addItem(textHL)
HLLegend = pg.ScatterPlotItem(size=10, pen=pg.mkPen('w'), pxMode=True)
walk_w.addItem(HLLegend)
HLLegend.addPoints([{'pos': [-3, -10], 'pen':pg.mkPen(color='#5566AA', width=2), 'brush': pg.mkBrush((0,0,0,0)), 'symbol': 's'}])

textHR = pg.TextItem('Hip Right')
textHR.setPos(-2.9, -10.3)
walk_w.addItem(textHR)
HRLegend = pg.ScatterPlotItem(size=10, pen=pg.mkPen('w'), pxMode=True)
walk_w.addItem(HRLegend)
HRLegend.addPoints([{'pos': [-3, -10.5], 'pen':pg.mkPen(color='#5566AA', width=2), 'brush': pg.mkBrush((0,0,0,0)), 'symbol': 'o'}])

textHL = pg.TextItem('Knee Left')
textHL.setPos(-0.5, -9.8)
walk_w.addItem(textHL)
HLLegend = pg.ScatterPlotItem(size=10, pen=pg.mkPen('w'), pxMode=True)
walk_w.addItem(HLLegend)
HLLegend.addPoints([{'pos': [-0.6, -10], 'pen':pg.mkPen(color='#8DC43C', width=2), 'brush': pg.mkBrush((0,0,0,0)), 'symbol': 's'}])

textHR = pg.TextItem('Knee Right')
textHR.setPos(-0.5, -10.3)
walk_w.addItem(textHR)
HRLegend = pg.ScatterPlotItem(size=10, pen=pg.mkPen('w'), pxMode=True)
walk_w.addItem(HRLegend)
HRLegend.addPoints([{'pos': [-0.6, -10.5], 'pen':pg.mkPen(color='#8DC43C', width=2), 'brush': pg.mkBrush((0,0,0,0)), 'symbol': 'o'}])

textHL = pg.TextItem('Ankle Left')
textHL.setPos(1.9, -9.8)
walk_w.addItem(textHL)
HLLegend = pg.ScatterPlotItem(size=10, pen=pg.mkPen('w'), pxMode=True)
walk_w.addItem(HLLegend)
HLLegend.addPoints([{'pos': [1.8, -10], 'pen':pg.mkPen(color='#CC3333', width=2), 'brush': pg.mkBrush((0,0,0,0)), 'symbol': 's'}])

textHR = pg.TextItem('Ankle Right')
textHR.setPos(1.9, -10.3)
walk_w.addItem(textHR)
HRLegend = pg.ScatterPlotItem(size=10, pen=pg.mkPen('w'), pxMode=True)
walk_w.addItem(HRLegend)
HRLegend.addPoints([{'pos': [1.8, -10.5], 'pen':pg.mkPen(color='#CC3333', width=2), 'brush': pg.mkBrush((0,0,0,0)), 'symbol': 'o'}])

#Light colors represent predicted results
textPre = pg.TextItem('Faint-colored curves/symbols represent prediction results\nDarker curves/sybmols represent the ground truth')
textPre.setPos(-3.15, -10.9)
walk_w.addItem(textPre)
# symbols 
# symbols on the left leg (ground turth)
gtHLSymbol = pg.ScatterPlotItem(size=10, pen=pg.mkPen('w'), pxMode=True)
walk_w.addItem(gtHLSymbol)
gtKLSymbol = pg.ScatterPlotItem(size=10, pen=pg.mkPen('w'), pxMode=True)
walk_w.addItem(gtKLSymbol)
gtALSymbol = pg.ScatterPlotItem(size=10, pen=pg.mkPen('w'), pxMode=True)
walk_w.addItem(gtALSymbol)
# symbols on the right leg (ground turth)
gtHRSymbol = pg.ScatterPlotItem(size=10, pen=pg.mkPen('w'), pxMode=True)
walk_w.addItem(gtHRSymbol)
gtKRSymbol = pg.ScatterPlotItem(size=10, pen=pg.mkPen('w'), pxMode=True)
walk_w.addItem(gtKRSymbol)
gtARSymbol = pg.ScatterPlotItem(size=10, pen=pg.mkPen('w'), pxMode=True)
walk_w.addItem(gtARSymbol)

# symbols on the left leg (predicted)
preHLSymbol = pg.ScatterPlotItem(size=10, pen=pg.mkPen('w'), pxMode=True)
walk_w.addItem(preHLSymbol)
preKLSymbol = pg.ScatterPlotItem(size=10, pen=pg.mkPen('w'), pxMode=True)
walk_w.addItem(preKLSymbol)
preALSymbol = pg.ScatterPlotItem(size=10, pen=pg.mkPen('w'), pxMode=True)
walk_w.addItem(preALSymbol)
# symbols on the right leg (predicted)
preHRSymbol = pg.ScatterPlotItem(size=10, pen=pg.mkPen('w'), pxMode=True)
walk_w.addItem(preHRSymbol)
preKRSymbol = pg.ScatterPlotItem(size=10, pen=pg.mkPen('w'), pxMode=True)
walk_w.addItem(preKRSymbol)
preARSymbol = pg.ScatterPlotItem(size=10, pen=pg.mkPen('w'), pxMode=True)
walk_w.addItem(preARSymbol)


# walk window setting
walk_w.disableAutoRange('xy')
walk_w.autoRange()

half_window_size = 300
i = 400

def update():
    global i, plotGL, plotGR, half_window_size

    gtHLsp.clear(), preHLsp.clear()
    gtKLsp.clear(), preKLsp.clear()
    gtALsp.clear(), preALsp.clear()
    gtHRsp.clear(), preHRsp.clear()
    gtKRsp.clear(), preKRsp.clear()
    gtARsp.clear(), preARsp.clear()
    gtHLSymbol.clear(), preHLSymbol.clear()
    gtKLSymbol.clear(), preKLSymbol.clear()
    gtALSymbol.clear(), preALSymbol.clear()
    gtHRSymbol.clear(), preHRSymbol.clear()
    gtKRSymbol.clear(), preKRSymbol.clear()
    gtARSymbol.clear(), preARSymbol.clear()

    # HL_w_data = gtHLdata[i-half_window_size:i+half_window_size] + 180
    # gt_KL_w_data = gtKLdata[i-half_window_size:i+half_window_size] + 90
    # gt_AL_w_data = gtALdata[i-half_window_size:i+half_window_size] + 40

    # HR_w_data = gtHRdata[i-half_window_size:i+half_window_size] - 40
    # gt_KR_w_data = gtKRdata[i-half_window_size:i+half_window_size] - 90
    # gt_AR_w_data = gtARdata[i-half_window_size:i+half_window_size] - 180

    # gt data
    gt_HL_w_data = gtHLdata[i-half_window_size:i+half_window_size] + 180
    gt_HR_w_data = gtHRdata[i-half_window_size:i+half_window_size] + 115
    gt_KL_w_data = gtKLdata[i-half_window_size:i+half_window_size] + 20
    gt_KR_w_data = gtKRdata[i-half_window_size:i+half_window_size] - 20
    gt_AL_w_data = gtALdata[i-half_window_size:i+half_window_size] - 115   
    gt_AR_w_data = gtARdata[i-half_window_size:i+half_window_size] - 180
    # predited data
    pre_HL_w_data = preHLdata[i-half_window_size:i+half_window_size] + 180
    pre_HR_w_data = preHRdata[i-half_window_size:i+half_window_size] + 115
    pre_KL_w_data = preKLdata[i-half_window_size:i+half_window_size] + 20
    pre_KR_w_data = preKRdata[i-half_window_size:i+half_window_size] - 20
    pre_AL_w_data = preALdata[i-half_window_size:i+half_window_size] - 115   
    pre_AR_w_data = preARdata[i-half_window_size:i+half_window_size] - 180

    # gt curves
    curveGtHL.setData(gt_HL_w_data)
    gtHLsp.addPoints([{'pos': [half_window_size, gt_HL_w_data[half_window_size]], 'pen':pg.mkPen(color='#5566AA', width=2), 'brush': pg.mkBrush((0,0,0,0)), 'symbol': 's'}])
    curveGtKL.setData(gt_KL_w_data)
    gtKLsp.addPoints([{'pos': [half_window_size, gt_KL_w_data[half_window_size]], 'pen':pg.mkPen(color='#8DC43C', width=2), 'brush': pg.mkBrush((0,0,0,0)), 'symbol': 's'}])
    curveGtAL.setData(gt_AL_w_data)
    gtALsp.addPoints([{'pos': [half_window_size, gt_AL_w_data[half_window_size]], 'pen':pg.mkPen(color='#CC3333', width=2), 'brush': pg.mkBrush((0,0,0,0)), 'symbol': 's'}])
    curveGtHR.setData(gt_HR_w_data)
    gtHRsp.addPoints([{'pos': [half_window_size, gt_HR_w_data[half_window_size]], 'pen':pg.mkPen(color='#5566AA', width=2), 'brush': pg.mkBrush((0,0,0,0)), 'symbol': 'o'}])
    curveGtKR.setData(gt_KR_w_data)
    gtKRsp.addPoints([{'pos': [half_window_size, gt_KR_w_data[half_window_size]], 'pen':pg.mkPen(color='#8DC43C', width=2), 'brush': pg.mkBrush((0,0,0,0)), 'symbol': 'o'}])
    curveGtAR.setData(gt_AR_w_data)
    gtARsp.addPoints([{'pos': [half_window_size, gt_AR_w_data[half_window_size]], 'pen':pg.mkPen(color='#CC3333', width=2), 'brush': pg.mkBrush((0,0,0,0)), 'symbol': 'o'}])

    # predicted curves
    curvePreHL.setData(pre_HL_w_data)
    preHLsp.addPoints([{'pos': [half_window_size, pre_HL_w_data[half_window_size]], 'pen':pg.mkPen(color='#5566AA88', width=2), 'brush': pg.mkBrush((0,0,0,0)), 'symbol': 's'}])
    curvePreKL.setData(pre_KL_w_data)
    preKLsp.addPoints([{'pos': [half_window_size, pre_KL_w_data[half_window_size]], 'pen':pg.mkPen(color='#8DC43C88', width=2), 'brush': pg.mkBrush((0,0,0,0)), 'symbol': 's'}])
    curvePreAL.setData(pre_AL_w_data)
    preALsp.addPoints([{'pos': [half_window_size, pre_AL_w_data[half_window_size]], 'pen':pg.mkPen(color='#CC333388', width=2), 'brush': pg.mkBrush((0,0,0,0)), 'symbol': 's'}])
    curvePreHR.setData(pre_HR_w_data)
    preHRsp.addPoints([{'pos': [half_window_size, pre_HR_w_data[half_window_size]], 'pen':pg.mkPen(color='#5566AA88', width=2), 'brush': pg.mkBrush((0,0,0,0)), 'symbol': 'o'}])
    curvePreKR.setData(pre_KR_w_data)
    preKRsp.addPoints([{'pos': [half_window_size, pre_KR_w_data[half_window_size]], 'pen':pg.mkPen(color='#8DC43C88', width=2), 'brush': pg.mkBrush((0,0,0,0)), 'symbol': 'o'}])
    curvePreAR.setData(pre_AR_w_data)
    preARsp.addPoints([{'pos': [half_window_size, pre_AR_w_data[half_window_size]], 'pen':pg.mkPen(color='#CC333388', width=2), 'brush': pg.mkBrush((0,0,0,0)), 'symbol': 'o'}])



    walk_w.removeItem(plotGL)
    walk_w.removeItem(plotGR)
    walk_w.removeItem(connectAL), curve_w.removeItem(connectALCurve)
    walk_w.removeItem(connectHL), curve_w.removeItem(connectHLCurve)
    walk_w.removeItem(connectKL), curve_w.removeItem(connectKLCurve)
    walk_w.removeItem(connectAR), curve_w.removeItem(connectARCurve)
    walk_w.removeItem(connectHR), curve_w.removeItem(connectHRCurve)
    walk_w.removeItem(connectKR), curve_w.removeItem(connectKRCurve)

    # ground truth leg and symbols
    gtHxL,gtHyL = getPositionOnCircle(0,0,4,gtHLdata[i])
    gtHxL_Sym,gtHyL_Sym = getPositionOnCircle(0,0,1,gtHLdata[i])
    gtKxL,gtKyL = getPositionOnCircle(gtHxL,gtHyL, 5, gtKLdata[i])
    gtKxLShort,gtKyLShort = getPositionOnCircle(gtHxL,gtHyL, 4.5, gtKLdata[i])
    gtAx1L,gtAy1L = getPositionOnCircle(gtKxL,gtKyL, 1, 270+gtALdata[i])
    gtAx2L,gtAy2L = getPositionOnCircle(gtKxL,gtKyL, 0.5, 270+gtALdata[i]+180)
    plotGL.setData(np.asarray([0,gtHxL,gtKxLShort,gtAx1L,gtAx2L,gtKxLShort]), np.asarray([0,gtHyL,gtKyLShort,gtAy1L,gtAy2L,gtKyLShort]), pen=pg.mkPen(width=1.5))
    gtHLSymbol.addPoints([{'pos': [gtHxL_Sym,gtHyL_Sym], 'pen':pg.mkPen(color='#5566AA', width=2), 'brush': pg.mkBrush((0,0,0,0)), 'symbol': 's'}])
    gtKLSymbol.addPoints([{'pos': [gtHxL, gtHyL], 'pen':pg.mkPen(color='#8DC43C', width=2), 'brush': pg.mkBrush((0,0,0,0)), 'symbol': 's'}])
    gtALSymbol.addPoints([{'pos': [gtAx1L, gtAy1L], 'pen':pg.mkPen(color='#CC3333', width=2), 'brush': pg.mkBrush((0,0,0,0)), 'symbol': 's'}])

    gtHxR,gtHyR = getPositionOnCircle(0,0,4,-gtHRdata[i])
    gtHxR_Sym,gtHyR_Sym = getPositionOnCircle(0,0,1,-gtHRdata[i])
    gtKxR,gtKyR = getPositionOnCircle(gtHxR,gtHyR, 5,-gtKRdata[i])
    gtKxRShort,gtKyRShort = getPositionOnCircle(gtHxR,gtHyR, 4.5,-gtKRdata[i])
    gtAx1R,gtAy1R = getPositionOnCircle(gtKxR,gtKyR, 1, 270-gtARdata[i])
    gtAx2R,gtAy2R = getPositionOnCircle(gtKxR,gtKyR, 0.5, 270-gtARdata[i]+180)
    plotGR.setData(np.asarray([0,gtHxR,gtKxRShort,gtAx1R,gtAx2R,gtKxRShort]), np.asarray([0,gtHyR,gtKyRShort,gtAy1R,gtAy2R,gtKyRShort]), pen=pg.mkPen(width=1.5))
    gtHRSymbol.addPoints([{'pos': [gtHxR_Sym,gtHyR_Sym], 'pen':pg.mkPen(color='#5566AA', width=2), 'brush': pg.mkBrush((0,0,0,0)), 'symbol': 'o'}])
    gtKRSymbol.addPoints([{'pos': [gtHxR, gtHyR], 'pen':pg.mkPen(color='#8DC43C', width=2), 'brush': pg.mkBrush((0,0,0,0)), 'symbol': 'o'}])
    gtARSymbol.addPoints([{'pos': [gtAx1R, gtAy1R], 'pen':pg.mkPen(color='#CC3333', width=2), 'brush': pg.mkBrush((0,0,0,0)), 'symbol': 'o'}])

    # predict symbols
    preHxL,preHyL = getPositionOnCircle(0,0,4,preHLdata[i])
    preHxL_Sym,preHyL_Sym = getPositionOnCircle(0,0,1,preHLdata[i])
    preKxL,preKyL = getPositionOnCircle(preHxL,preHyL, 5, preKLdata[i])
    preKxLShort,preKyLShort = getPositionOnCircle(preHxL,preHyL, 4.5, preKLdata[i])
    preAx1L,preAy1L = getPositionOnCircle(gtKxL,gtKyL, 1, 270+preALdata[i])
    preHLSymbol.addPoints([{'pos': [preHxL_Sym,preHyL_Sym], 'pen':pg.mkPen(color='#5566AA88', width=2), 'brush': pg.mkBrush((0,0,0,0)), 'symbol': 's'}])
    preKLSymbol.addPoints([{'pos': [preHxL, preHyL], 'pen':pg.mkPen(color='#8DC43C88', width=2), 'brush': pg.mkBrush((0,0,0,0)), 'symbol': 's'}])
    preALSymbol.addPoints([{'pos': [preAx1L,preAy1L], 'pen':pg.mkPen(color='#CC333388', width=2), 'brush': pg.mkBrush((0,0,0,0)), 'symbol': 's'}])
    
    connectAL.setData(np.asarray([gtAx1L, preAx1L]), np.asarray([gtAy1L, preAy1L]), pen=pg.mkPen(width=1.5, style=QtCore.Qt.DotLine))
    connectHL.setData(np.asarray([gtHxL_Sym, preHxL_Sym]), np.asarray([gtHyL_Sym, preHyL_Sym]), pen=pg.mkPen(width=1.5, style=QtCore.Qt.DotLine))
    connectKL.setData(np.asarray([gtHxL, preHxL]), np.asarray([gtHyL, preHyL]), pen=pg.mkPen(width=1.5, style=QtCore.Qt.DotLine))

    preHxR,preHyR = getPositionOnCircle(0,0,4,-preHRdata[i])
    preHxR_Sym,preHyR_Sym = getPositionOnCircle(0,0,1,-preHRdata[i])
    preKxR,preKyR = getPositionOnCircle(gtHxR,gtHyR, 5,-gtKRdata[i])
    preKxRShort,preKyRShort = getPositionOnCircle(preHxR,preHyR, 4.5,-preKRdata[i])
    preAx1R,preAy1R = getPositionOnCircle(gtKxR,gtKyR, 1, 270-preARdata[i])
    preHRSymbol.addPoints([{'pos': [preHxR_Sym,preHyR_Sym], 'pen':pg.mkPen(color='#5566AA88', width=2), 'brush': pg.mkBrush((0,0,0,0)), 'symbol': 'o'}])
    preKRSymbol.addPoints([{'pos': [preHxR, preHyR], 'pen':pg.mkPen(color='#8DC43C88', width=2), 'brush': pg.mkBrush((0,0,0,0)), 'symbol': 'o'}])
    preARSymbol.addPoints([{'pos': [preAx1R, preAy1R], 'pen':pg.mkPen(color='#CC333388', width=2), 'brush': pg.mkBrush((0,0,0,0)), 'symbol': 'o'}])
    
    connectAR.setData(np.asarray([gtAx1R, preAx1R]), np.asarray([gtAy1R, preAy1R]), pen=pg.mkPen(width=1.5, style=QtCore.Qt.DotLine))
    connectHR.setData(np.asarray([gtHxR_Sym, preHxR_Sym]), np.asarray([gtHyR_Sym, preHyR_Sym]), pen=pg.mkPen(width=1.5, style=QtCore.Qt.DotLine))
    connectKR.setData(np.asarray([gtHxR, preHxR]), np.asarray([gtHyR, preHyR]), pen=pg.mkPen(width=1.5, style=QtCore.Qt.DotLine))

    connectHLCurve.setData(np.asarray([half_window_size, half_window_size]), np.asarray([gt_HL_w_data[half_window_size], pre_HL_w_data[half_window_size]]), pen=pg.mkPen(width=1.5, style=QtCore.Qt.DotLine))
    connectHRCurve.setData(np.asarray([half_window_size, half_window_size]), np.asarray([gt_HR_w_data[half_window_size], pre_HR_w_data[half_window_size]]), pen=pg.mkPen(width=1.5, style=QtCore.Qt.DotLine))
    connectKLCurve.setData(np.asarray([half_window_size, half_window_size]), np.asarray([gt_KL_w_data[half_window_size], pre_KL_w_data[half_window_size]]), pen=pg.mkPen(width=1.5, style=QtCore.Qt.DotLine))
    connectKRCurve.setData(np.asarray([half_window_size, half_window_size]), np.asarray([gt_KR_w_data[half_window_size], pre_KR_w_data[half_window_size]]), pen=pg.mkPen(width=1.5, style=QtCore.Qt.DotLine))
    connectALCurve.setData(np.asarray([half_window_size, half_window_size]), np.asarray([gt_AL_w_data[half_window_size], pre_AL_w_data[half_window_size]]), pen=pg.mkPen(width=1.5, style=QtCore.Qt.DotLine))
    connectARCurve.setData(np.asarray([half_window_size, half_window_size]), np.asarray([gt_AR_w_data[half_window_size], pre_AR_w_data[half_window_size]]), pen=pg.mkPen(width=1.5, style=QtCore.Qt.DotLine))

    walk_w.addItem(plotGL)
    walk_w.addItem(plotGR)
    walk_w.addItem(connectAL), curve_w.addItem(connectALCurve)
    walk_w.addItem(connectHL), curve_w.addItem(connectHLCurve)
    walk_w.addItem(connectKL), curve_w.addItem(connectKLCurve)
    walk_w.addItem(connectAR), curve_w.addItem(connectARCurve)
    walk_w.addItem(connectHR), curve_w.addItem(connectHRCurve)
    walk_w.addItem(connectKR), curve_w.addItem(connectKRCurve)

    i+=5
        
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
