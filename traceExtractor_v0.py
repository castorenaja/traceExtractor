#!/usr/bin/python

import Tkinter, tkFileDialog
import os.path
import numpy as np
#from numpy import *
import matplotlib.pyplot as plt
#import matplotlib as mpl
#from numpy import diff
#import cv2
#from scipy import stats
from math import factorial
import pandas

class TkAPAnalyzerApp(Tkinter.Tk):
    
    def __init__(self, parent):
        Tkinter.Tk.__init__(self, parent)
        self.parent = parent
        self.filename = None
        self.resizable(True,False)
        w, h = self.winfo_screenwidth(), self.winfo_screenheight()
        ch = 290
        self.geometry("%dx%d+0+0" % (0.99*w, ch))
##        self.geometry('1275x360')
        self.GUIbuilder()
    

    def GUIbuilder(self):

        FrameUno = Tkinter.Frame(self, bd=3, relief='solid')
        FrameUno.grid(column=0,columnspan=6,row=0,sticky='EW')

        Tkinter.Label(FrameUno,text="dataExtractor: Data Processing Tool", font=('Helvetica Neue',13,'bold'),
                      anchor="c",fg="white",bg="blue").grid(column=0,columnspan=6,row=0,sticky= 'EW',pady=10,padx=10)

        browseBttn = Tkinter.Button(FrameUno, text='Browser')
        browseBttn.grid(column=0,columnspan=1,row=1,pady=10,padx=20)
        browseBttn.configure(command = self.runBrowser)

        self.fileName = Tkinter.StringVar()
        self.fileName.set("Please Select File")
        fileLabel = Tkinter.Label(FrameUno, textvariable=self.fileName)
        fileLabel.grid(column=1,columnspan=2,row=1,sticky="W",pady=10,padx=20)
        
        self.checkAcq = Tkinter.IntVar()
        self.checkAcq.set(0)
        self.acqCB = Tkinter.Checkbutton(FrameUno, text="Use Custom Acquisition Speed", anchor ="w",variable=self.checkAcq)
        self.acqCB.grid(column=3,row=1, sticky='E')        

        Tkinter.Label(FrameUno,text="Acquisiton Speed (kHz):",anchor="c",fg="white",bg="black").grid(column=4,columnspan=1,row=1,sticky='EW',padx=10,pady=10)
        self.iAcqSpeed = Tkinter.DoubleVar()
        self.iAcqSpeed.set(1)
        self.iAcqSpeedEntry = Tkinter.Entry(FrameUno,textvariable=self.iAcqSpeed,width=30)
        self.iAcqSpeedEntry.grid(column=5,columnspan=1,row=1,sticky='EW',padx=10)  

        Tkinter.Label(FrameUno,text="Filtering Option:",anchor="c",fg="white",bg="black").grid(column=0,columnspan=1,row=2,sticky='EW',padx=10,pady=10)
        self.spinFreq = Tkinter.StringVar()
        self.fSBox = Tkinter.Spinbox(FrameUno, values=('No Filter','Opt1 (7)','Opt2 (31)','Opt3 (91)','Opt4 (151)','Opt5 (551)'),textvariable=self.spinFreq)
        self.spinFreq.set('No Filter')
        self.fSBox.grid(column=1,columnspan=1,row=2, sticky='EW',padx=25)

        self.checkCustomFlt = Tkinter.IntVar()
        self.checkCustomFlt.set(0)
        self.fltCB = Tkinter.Checkbutton(FrameUno, text="Custom Filtering", bg="gray", anchor ="c",variable=self.checkCustomFlt)
        self.fltCB.grid(column=2,columnspan=1,row=2, sticky='EW',padx=10,pady=10)

        self.customFlt = Tkinter.DoubleVar()
        self.customFltEntry = Tkinter.Entry(FrameUno,textvariable=self.customFlt,width=30)
        self.customFltEntry.grid(column=3,columnspan=1,row=2,sticky='EW',padx=10)

        Tkinter.Label(FrameUno,text="Offset:",anchor="c",fg="white",bg="black").grid(column=4,row=2,sticky='EW',padx=10,pady=10)
        self.iOffset = Tkinter.DoubleVar()
        self.iOffsetEntry = Tkinter.Entry(FrameUno,textvariable=self.iOffset,width=30)
        self.iOffsetEntry.grid(column=5,columnspan=1,row=2,sticky='EW',padx=10)       

        Tkinter.Label(FrameUno,text="Define ROI?",anchor="c",fg="white",bg="black").grid(column=0,row=3,sticky='EW',padx=10,pady=10)

        self.checkROI = Tkinter.IntVar()
        self.checkROI.set(0)
        self.roiCB = Tkinter.Checkbutton(FrameUno, text="Yes (Check)", anchor ="w",variable=self.checkROI)
        self.roiCB.grid(column=1,row=3, sticky='EW')
        
        Tkinter.Label(FrameUno,text="From:",anchor="c",fg="white",bg="gray45").grid(column=2,columnspan=1,row=3,sticky= 'EW',pady=5,padx=10)
        self.iLimit = Tkinter.DoubleVar()
        self.iLimitEntry = Tkinter.Entry(FrameUno,textvariable=self.iLimit,width=30)
        self.iLimitEntry.grid(column=3,columnspan=1,row=3,sticky='EW',padx=10)

        Tkinter.Label(FrameUno,text="To:",anchor="c",fg="white",bg="gray45").grid(column=4,columnspan=1,row=3,sticky= 'EW',pady=5,padx=10)
        self.sLimit = Tkinter.DoubleVar()
        self.sLimitEntry = Tkinter.Entry(FrameUno,textvariable=self.sLimit,width=30)
        self.sLimitEntry.grid(column=5,columnspan=1,row=3,sticky='EW',padx=10)

        self.runButton = Tkinter.Button(FrameUno, state="disabled", text ="RUN",fg='red',font=('Helvetica Neue',16,'bold'))
        self.runButton.grid(column=2,columnspan=1,row=5,pady=30,padx=20)
        self.runButton.configure(command = self.runAnalysis)

#        self.exportButton = Tkinter.Button(FrameUno, state="disabled", text ="EXPORT DATA",fg='red',font=('Helvetica Neue',16,'bold'))
#        self.exportButton.grid(column=3,columnspan=1,row=5,pady=30,padx=20)
#        self.exportButton.configure(command = self.runAnalysis)

        self.checkExport = Tkinter.IntVar()
        self.checkExport.set(0)
        self.exportCB = Tkinter.Checkbutton(FrameUno, state = "disabled", text="EXPORT DATA", anchor ="w",variable=self.checkExport,font=('Helvetica Neue',12,'bold'),command=self.enableExport)
        self.exportCB.grid(column=3,row=5, sticky='W')
        
        self.checkNormExport = Tkinter.IntVar()
        self.checkNormExport.set(0)
        self.ntimeCB = Tkinter.Checkbutton(FrameUno, state = "disabled", text="EXPORT DATA (Initialized Time)", anchor ="w",variable=self.checkNormExport,font=('Helvetica Neue',12,'bold'),command=self.enableNormExport)
        self.ntimeCB.grid(column=4,row=5, sticky='W') 

        FrameUno.columnconfigure(0,weight=1,uniform='a')
        FrameUno.columnconfigure(1,weight=1,uniform='a')
        FrameUno.columnconfigure(2,weight=1,uniform='a')
        FrameUno.columnconfigure(3,weight=1,uniform='a')
        FrameUno.columnconfigure(4,weight=1,uniform='a')
        FrameUno.columnconfigure(5,weight=1,uniform='a')
        
        self.update()
        self.geometry(self.geometry())
        self.columnconfigure(0,weight=1)
        self.columnconfigure(1,weight=1)
        FrameUno.columnconfigure(0,weight=1)


    def enableExport(self):
        if self.checkExport.get():
            self.checkNormExport.set(0)
            
    def enableNormExport(self):
        if self.checkNormExport.get():
            self.checkExport.set(0)


            
    def runBrowser(self):

        self.file_opt = options = {}
        options['defaultextension'] = '.txt'
        options['filetypes'] = [('Text Files', '.txt')]
##        options['initialdir'] = 'C:\\'
        options['title'] = 'Select File Containing AP Data (Time and Voltage)'

        self.fileName.set(tkFileDialog.askopenfilename(**self.file_opt))    

        if(os.path.isfile(self.fileName.get())):
            self.runButton.configure(state = "normal")
            self.exportCB.configure(state = "normal")
            self.ntimeCB.configure(state = "normal")
#            self.exportButton.configure(state = "normal")

    def runAnalysis(self):

        fiLimit = self.iLimit.get()
        fsLimit = self.sLimit.get()

        intCheckOpt = 0
        if self.checkROI.get():
            intCheckOpt = 1

        if(os.path.isfile(self.fileName.get())):
            self.mainToolFunction(self.fileName.get(),intCheckOpt,fiLimit,fsLimit)  


    def savitzky_golay(self, y, window_size, order, deriv=0, rate=1):
        r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
        The Savitzky-Golay filter removes high frequency noise from data.
        It has the advantage of preserving the original shape and
        features of the signal better than other types of filtering
        approaches, such as moving averages techniques.
        Parameters
        ----------
        y : array_like, shape (N,)
            the values of the time history of the signal.
        window_size : int
            the length of the window. Must be an odd integer number.
        order : int
            the order of the polynomial used in the filtering.
            Must be less then `window_size` - 1.
        deriv: int
            the order of the derivative to compute (default = 0 means only smoothing)
        Returns
        -------
        ys : ndarray, shape (N)
            the smoothed signal (or it's n-th derivative).
        Notes
        -----
        The Savitzky-Golay is a type of low-pass filter, particularly
        suited for smoothing noisy data. The main idea behind this
        approach is to make for each point a least-square fit with a
        polynomial of high order over a odd-sized window centered at
        the point.
        Examples
        --------
        t = np.linspace(-4, 4, 500)
        y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
        ysg = savitzky_golay(y, window_size=31, order=4)
        import matplotlib.pyplot as plt
        plt.plot(t, y, label='Noisy signal')
        plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
        plt.plot(t, ysg, 'r', label='Filtered signal')
        plt.legend()
        plt.show()
        References
        ----------
        .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
           Data by Simplified Least Squares Procedures. Analytical
           Chemistry, 1964, 36 (8), pp 1627-1639.
        .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
           W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
           Cambridge University Press ISBN-13: 9780521880688
        """
    ##    import numpy as np
    ##    from math import factorial
        
        try:
            window_size = np.abs(np.int(window_size))
            order = np.abs(np.int(order))
        except ValueError, msg:
            raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("window_size is too small for the polynomials order")
        order_range = range(order+1)
        half_window = (window_size -1) // 2
        # precompute coefficients
        b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
        m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
        # pad the signal at the extremes with
        # values taken from the signal itself
        firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
        lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
       
        return np.convolve( m[::-1], y, mode='valid')

    # =======================================================================================#


    def mainToolFunction(self,myFile,checkROI,mnTime,mxTime): # MAIN LOOP AND FUNCTION CALLING ===================================#

        global inFile, Time, myVoltage, dt, minTime, maxTime, roiOpt

        roiOpt = checkROI

        inFile = myFile          

        cnames = ('Time','LPressure','RPressure','Servonull','Diameter','OD','Temp')
        dataFr = pandas.read_csv(inFile, delim_whitespace=True, names = cnames, skiprows = 3, dtype = object)
        dataFr = dataFr.drop(dataFr.index[-1])
        dataFr = dataFr.astype(float)
                
        Time = dataFr['Time'].values
        myVoltage = dataFr['Diameter'].values
        
        global fZero, cid, fCoeff, fOrder, fOpt

        fOrder = 5

        fOpt = self.spinFreq.get()
        
        if fOpt == 'No Filter':
            fCoeff = 0
        if fOpt == 'Opt1 (7)':
            fCoeff = 7
        if fOpt == 'Opt2 (31)':
            fCoeff = 31
        if fOpt == 'Opt3 (91)':
            fCoeff = 91
        if fOpt == 'Opt4 (151)':
            fCoeff = 151
        if fOpt == 'Opt5 (551)':
            fCoeff = 551
       

        cbCustomFlt = 0
        if self.checkCustomFlt.get():
            cbCustomFlt = 1

        if cbCustomFlt == 0:
            if fOpt == 'No Filter':
                pass
            else:
                myVoltage = self.savitzky_golay(myVoltage, fCoeff, fOrder)
                dataFr.Diameter = myVoltage
        if cbCustomFlt == 1:
            fCoeff = int(self.customFlt.get())
            if (fCoeff % 2 == 0):
                fCoeff = fCoeff-1               
            else:
                pass
            myVoltage = self.savitzky_golay(myVoltage, fCoeff, fOrder)
            dataFr.Diameter = myVoltage




        NumDataPts = int(Time.shape[0])

        if self.checkAcq.get():
            del Time
            dt = 1./(self.iAcqSpeed.get())
            Time = np.arange(0.,NumDataPts*dt,dt,dtype=float)

        if checkROI == 1:
            minTime = mnTime
            maxTime = mxTime
            minIndex = np.abs(Time-minTime).argmin()
            maxIndex = np.abs(Time-maxTime).argmin()
            Time = Time[minIndex:maxIndex]
            myVoltage = myVoltage[minIndex:maxIndex]
            
        if checkROI == 0:
            minTime = np.amin(Time)
            maxTime = np.amax(Time)
          
        myVoltage = myVoltage-self.iOffset.get()


        fZero = plt.figure(figsize=(16,8))

        plt.plot(Time, myVoltage, 'k-')
                           
        plt.xlabel('Time (minutes)')
        plt.ylabel('Diameter ($\mu$m)')
        
        if checkROI == 1:
            plt.xlim(minTime,maxTime)
            
        if checkROI == 0:
            plt.xlim(np.amin(Time), np.amin(Time)+((np.amax(Time)-np.amin(Time))*1))

        export = 0

        if self.checkExport.get():
            export = 1
        if self.checkNormExport.get():
            export = 1


        if export == 1:
            dataFile = inFile[:-4]+'_extractedData.xlsx'
    
            fileHeader = ['Time','Diameter']
            fileUnits = {'Time':'minutes','Diameter':'um'}
            sdataFr = dataFr[['Time','Diameter']]
            
            if self.checkROI.get():
                sdataFr = sdataFr[minIndex:maxIndex]
                if self.checkNormExport.get():
                    sdataFr.Time = sdataFr.Time-np.amin(Time)
    
            myDF = pandas.DataFrame(columns=fileHeader)
            myDF = myDF.append(fileUnits, ignore_index=True)
            myDF = myDF.append(sdataFr, ignore_index=True)
            
            myWriter = pandas.ExcelWriter(dataFile, engine='xlsxwriter')
            myDF.to_excel(myWriter,index=False,sheet_name='ExtractedData')
            myWorkbook = myWriter.book
            myWorksheet = myWriter.sheets['ExtractedData']
            myWorksheet.set_zoom(90)
            myWorksheet.set_column('A:B',12)
            myWorkbook.add_format({'align':'center','valign':'vcenter'})
            myWriter.save()

        plt.show()



if __name__ == "__main__":
    app = TkAPAnalyzerApp(None)
    app.title('dataExtractor_v0 by Jorge Castorena')
    app.mainloop()
    


