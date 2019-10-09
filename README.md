# GeneralizedSketchFamilies
Code for Generalized Sketch Families for Network Traffic Measurement (Sigmetrics 2020) including flow size/spread measurement using cSketch, bSketch and vSketch with plug-in unit estimators (counter, bitmap, FM sketch and HyperLogLog).

An example of data traces that are accepted is given in /data and the measurement results are given in /result.

CPU: 
    1.GeneralSketch floder contains code for flow size/spread measurement for  
      bSketch(GeneralSketchBloom.java)/cSketch(GeneralCountMin.java)/vSketch(GeneralvSkt.java). 
       
     2.GeneralUnion floder contains code for Spitial Join Experiment. The input data should be under data/union/. splitlni means data for        joining of i data trace and outputiv.txt means ith input data trace. Statistic results srcdstsize.txt and dstspread.txt should be 
       under data/union. The format should be same as the ones under data/. 
       
     3.GeneralTime floder contains code for Time Join Experiment. The input data should be under data/time/. outputiv.txt means ith input 
       data tracedata for time join. data/time/Ti/ contains statistic result for first i data traces. The statistic result name is  
       outputsrcDstCount.txt(same format as srcdstsize.txt),outputdstCard.txt(same format as dstspread.txt).
       
     4.all result file will under floder result/.

FPGA:The two .zip file contains whole projects for bSketches and vSketches under xilinx vivado. The board we use is xilinx nexys A7-100T 
     development board. 
     
GPU:The project under microsoft visual studio using cuda 10.0. Please change setting array in newsketch.cuh to change type.
    
OVS: datapath folder in openvswitch-2.10.1.
