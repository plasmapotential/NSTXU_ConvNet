# FluxImport2.py

# Date:         20180315
# Description:  ANSYS ACT Extension for adding heat fluxes from CSVs to Tile Faces
# Engineer:     Tom Looby
# Project:      PPPL NSTX-U Recovery

import os
import datetime
clr.AddReference("Ans.UI.Toolkit")
clr.AddReference("Ans.UI.Toolkit.Base")
from Ansys.UI.Toolkit import *
import units
import graphics
import time



# Start Logging.  Lives in <working_directory>_files\dp0\SYS\MECH
def init(context):
    ExtAPI.Log.WriteMessage("\n\n===== Flux Profile Importer ACT Script Initialized... =====\n\n")

def OnClickB1(analysis):    
    
    #Start a clock for timestamping
    t0 = time.time()
    
    for file_iter in range(0,99):
       # Print Stuff to Log File
       t1 = time.time()
       ExtAPI.Log.WriteMessage(" ")
       ExtAPI.Log.WriteMessage("=============================================================")
       ExtAPI.Log.WriteMessage("Iteration Number: {:0>6}".format(file_iter+1)) 
       ExtAPI.Log.WriteMessage("Time Elapsed at Iteration Start [s]: {:f}".format(t1 - t0))
       ExtAPI.Log.WriteMessage(" ")
       
       #See if there are any heat fluxes from last time and delete them
       child_count = ExtAPI.DataModel.Project.Model.Analyses[0].Children.Count
       if child_count > 3:
          for del_iter in range(3,child_count):
             ExtAPI.DataModel.Project.Model.Analyses[0].Children[2].Delete()
       
       #Set up file IO for this iteration through loop       
       infile = (r'C:\Users\thoma\Documents\school\grad\masters_thesis\NSTX\flux_import\input_data\tensorflow\flux_profile_{:0>6}.txt'.format(file_iter+1))
       outfile = (r'C:\Users\thoma\Documents\school\grad\masters_thesis\NSTX\flux_import\input_data\tensorflow\TC_profile_{:0>6}.txt'.format(file_iter+1))
       ExtAPI.Log.WriteMessage("Input File:")
       ExtAPI.Log.WriteMessage(infile)
       ExtAPI.Log.WriteMessage("Output File:")
       ExtAPI.Log.WriteMessage(outfile)
       
       #Basic Initialization Stuff
       model = ExtAPI.DataModel.Project.Model
       part1 = model.Geometry.Children[0]
       body1 = part1.Children[0]
       face_id = []
       face_area = []
       face_ctr = []
       face_data = []
       face_arr = []
       facecounter = 0
#==========================================================================
#                              Import Flux Data
#==========================================================================
      
       # Import Data into data array.  Should be comma delimited.  fh1 opens CSV
       fh1 = open(infile)
       data = []
       for line in fh1:
          li=line.strip()
          if not li.startswith("#"):
             data.append(li.split(', '))
       fh1.close
      
       #Number of Time Steps
       length = len(data)
       #Number of Spatial Steps + 1
       length2 = len(data[0])
       #Number of spatial steps
       n = length2 - 1
       # Tile Width at narrowest point [m]
       width = 0.02242
       # Length Overall [m]
       loa = 0.15431
       # Area [m^2]
       n_area = width*(loa/n)
       ExtAPI.Log.WriteMessage("Area per Slice: %.7f" % (n_area,))
      
       # Sometimes weird garbage faces are created when slicing: ignore them.
       # Also, sometimes if a face is too small ANSYS complains about
       # applying a load to it.  This throws a comment into the log file. 
       if n_area < 0.000005:
          ExtAPI.Log.WriteMessage("Note!!!  Any slices with area < 5E-6m^2 are ignored!")
          ExtAPI.Log.WriteMessage("ANSYS cannot apply loads to arbitrarily small surfaces.")
          ExtAPI.Log.WriteMessage("Try increasing the number of slices.")
       # Build Time Array
       time_mag = "[ "
       for i in range(length - 1):
          time_mag += ("Quantity(\"%s [s]\"), " % (data[i][0],))
       time_mag += ("Quantity(\"%s [s]\") ]" % (data[length - 1][0],))
      
       # Build Flux Magnitude Array
       flux_mag = ["[ " for x in range(length2-1)]
       i=0
       col_sum = [0]*(length2-1)
       for i in range(length-1):
          j=0
          for j in range(length2 - 1):
             flux_mag[j] += ("Quantity(\"%s [W/m^2]\"), " % (data[i][j+1],))
             col_sum[j] = float(data[i][j+1]) + col_sum[j]
             j=j+1
          i=i+1
       j=0
      
       for j in range(length2-1):
          flux_mag[j] += ("Quantity(\"%s [W/m^2]\") ]" % (data[length-1][j+1],))
          j=j+1
       
#==========================================================================
#                              Add Heat Flux
#==========================================================================
      
       # Get data for top surface faces
       for index, face in enumerate(body1.GetGeoBody().Faces, start=0):
          centroid = face.Centroid
          if centroid[1] > 0.053:
             face_data.append([face.Id, face.Area, face.Centroid[0], face.Centroid[1], face.Centroid[2]])          
      
       # Sort the faces in the Z direction (radially)
       face_data.sort(key=lambda face_arr: face_arr[4])
       
       #Assign the model we are working on to variable
       model = ExtAPI.DataModel.Project.Model
       
       i=0
       slice = 0
       j=0
       #Loop through face data and add heat fluxes as required
       for index, face_arr in enumerate(face_data):
          #For troubleshooting...   
          #ExtAPI.Log.WriteMessage("INDEX: %i, SLICE: %i" % (index, slice))
      
          # Sometimes weird garbage faces are created when slicing: ignore them.
          # Also, sometimes if a face is too small ANSYS complains about
          # applying a load to it.
          if face_arr[1] < 0 or face_arr[1] < 0.01*n_area or face_arr[1] < 0.000005:
             continue
          
          # If no flux is on slice for all time, bail!
          elif col_sum[slice] == 0:
            pass       
             
          # Apply heat flux per conditions below     
          else:
             flux = model.Analyses[0].AddHeatFlux()
             # Create Empty Selection
             selection = ExtAPI.SelectionManager.CreateSelectionInfo(SelectionTypeEnum.GeometryEntities)
             #Assign Flux Location by Geometric ID
             selection.Entities =  [ExtAPI.DataModel.GeoData.GeoEntityById(face_data[index][0])]
             flux.Location = selection
             #Build Commands from CSV Data
             time_command = "flux.Magnitude.Inputs[0].DiscreteValues = %s" % (time_mag,)
             mag_command = "flux.Magnitude.Output.DiscreteValues = %s" % (flux_mag[slice],)
             #Execute Commands to Create Fluxes
             exec(time_command)
             exec(mag_command)
             # ExtAPI.Log.WriteMessage("INDEX: %i  ID: %d  ZCoord: %.7f  Area: %.7f" % (index, face_arr[0], face_arr[4], face_arr[1]))
      
          # Check for multi-castellation face and act accordingly
          if j < 1:
             slice_area = face_arr[1]
          else:
             slice_area = face_arr[1] + slice_area
          if slice_area < n_area*0.9:
             j = j+1
          else:
             slice = slice + 1
             j=0
      
          if slice > length2 - 1:
             break
          
       
       # For reference, this is format of python ACT command to create flux table
       #flux.Magnitude.Inputs[0].DiscreteValues = [ Quantity("0 [s]"), Quantity("5 [s]"), Quantity("6 [s]")]
       #flux.Magnitude.Output.DiscreteValues = [Quantity("10000000 [W/m^2]"), Quantity("10000000 [W/m^2]"), Quantity("0 [W/m^2]")]
       
       # To print to a file for error checking
       #outfile = open(r'C:\Users\thoma\Desktop\logger.txt','w')
       #outfile.write(command1)
       #outfile.close()
      
       #Solve The Model
       ExtAPI.DataModel.Project.Model.Solve(1)
	   
       #====This is for creating an array "tc_arr" with all TC data for all time
       #  This method takes forever, but it works until a better method is found
       T1 = ExtAPI.DataModel.Project.Model.Analyses[0].Solution.Children[1]
       T2 = ExtAPI.DataModel.Project.Model.Analyses[0].Solution.Children[2]
       T3 = ExtAPI.DataModel.Project.Model.Analyses[0].Solution.Children[3]
       T4 = ExtAPI.DataModel.Project.Model.Analyses[0].Solution.Children[4]
       T5 = ExtAPI.DataModel.Project.Model.Analyses[0].Solution.Children[5]
       tc_arr = []
       # Fill tc_arr with time evolving TC data
       for i in range (length-1):
          disp = "%f [s]" % float(data[i][0])
          T1.DisplayTime =Quantity (disp) 
          #T1.EvaluateAllResults()
          
          T2.DisplayTime =Quantity (disp) 
          #T2.EvaluateAllResults()
          
          T3.DisplayTime =Quantity (disp) 
          #T3.EvaluateAllResults()
          
          T4.DisplayTime =Quantity (disp) 
          #T4.EvaluateAllResults()
          
          T5.DisplayTime =Quantity (disp) 
          #T5.EvaluateAllResults()
          
          ExtAPI.DataModel.Project.Model.Analyses[0].Solution.EvaluateAllResults()
          
          tc_arr.append([T1.Temperature, T2.Temperature, T3.Temperature, T4.Temperature, T5.Temperature])
       
       #Write Data to output file for TensorFlow
       fh_out = open(outfile, "w")
       for item in tc_arr:
          fh_out.write("%s\n" % item)
       fh_out.close()
       
       #Write To Log File
       ExtAPI.Log.WriteMessage("Output Written")
       t2 = time.time()
       ExtAPI.Log.WriteMessage("Iteration Time [s]: {:f}".format(t2 - t1))
       ExtAPI.Log.WriteMessage(" ")
       ExtAPI.Log.WriteMessage("=============================================================")
       ExtAPI.Log.WriteMessage(" ")
       #====
    ExtAPI.Log.WriteMessage("All Flux Profiles Crunched...")
    ExtAPI.Log.WriteMessage("Total Time Elapsed [s]: {:f}".format(t2 - t0))
    ExtAPI.Log.WriteMessage("Script Exiting...")
    ExtAPI.Log.WriteMessage(" ")  
    
    
def LogButtonClicked(toolbarId, buttonId, analysis):
    now = datetime.datetime.now()
    outFile = SetUserOutput("ExtToolbarSample.log", analysis)
    f = open(outFile,'a')    
    f.write("*.*.*.*.*.*.*.*\n")
    f.write(str(now)+"\n")
    f.write("Toolbar "+toolbarId.ToString()+" - Button "+buttonId.ToString()+" Clicked. \n")
    f.write("*.*.*.*.*.*.*.*\n")
    f.close()    
    MessageBox.Show("Toolbar "+toolbarId.ToString()+" - Button "+buttonId.ToString()+" Clicked.") 
    
def SetUserOutput(filename, analysis):
    solverDir = analysis.WorkingDir
    return os.path.join(solverDir,filename)
