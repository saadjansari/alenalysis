import numpy as np
import pdb
from pathlib import Path
from DataHandler import FilamentSeries, CrosslinkerSeries
from read_config import *
# import vtk


def read_sim(spath, conf='U'):
    # Read Simulation into Series

    # read config files
    # print('\tReading yaml file...')
    # box_size = get_boxsize_confined(spath, conf)
    # time_snap = get_timeSnap(spath)
    # kappa = get_spring_constant(spath, idx=0)
    # rest_length = get_rest_length(spath, idx=0)
    # kT = get_kT(spath)
    # diameter_fil = get_diameter_sylinder(spath)
    # geometry = get_confining_geometry(spath)
    config = get_config(spath)

    # find data files of sim
    files_sylinder, files_protein, files_constraint = find_all_frames(spath / 'result')
    print('Reading {0} frames...'.format(len(files_sylinder)))
   
    # Read all files, and
    # Instantiate handler objects

    # Filaments
    pos_minus, pos_plus, orientation, gid = read_all_sylinder(files_sylinder)
    # FData = FilamentSeries(gid, pos_minus, pos_plus, orientation, box_size, time_snap, kT, diameter_fil)
    FData = FilamentSeries(gid, pos_minus, pos_plus, orientation, config)

    # Crosslinkers
    gid, pos0, pos1, link0, link1 = read_all_protein(files_protein)
    # XData = CrosslinkerSeries(gid, pos0, pos1, link0, link1, box_size, time_snap, kT, kappa, rest_length)
    XData = CrosslinkerSeries(gid, pos0, pos1, link0, link1, config)

    # Constraints
    # stress_uni, stress_bi = read_all_constraint(files_constraint)
    # pdb.set_trace()
    # FData = 1
    # XData = 1

    return FData, XData


def find_all_frames(spath):
    def fileKey(f):
        return int( f.parts[-1].split('_')[-1].split('.dat')[0] )
    def fileKey_c(f):
        return int( f.parts[-1].split('_')[-1].split('.pvtp')[0] )
    file_s = sorted( list(spath.glob('**/SylinderAscii*.dat')), key=fileKey)
    file_p = sorted( list(spath.glob('**/ProteinAscii*.dat')), key=fileKey)
    file_c = sorted( list(spath.glob('**/ConBlock*.pvtp')), key=fileKey_c)
    return file_s, file_p, file_c

def read_all_sylinder(file_list):

    # get filament number
    numS = count_sylinders(file_list[0])

    # initialize arrays for storing information
    pos_minus = np.zeros( (3, numS, len(file_list)))
    pos_plus = np.zeros( (3, numS, len(file_list)))
    orientation = np.zeros( (3, numS, len(file_list)))
    gid = np.zeros( (numS, len(file_list)))

    for idx,fil in enumerate(file_list):
        df = read_dat_sylinder(fil)
        gid[:,idx] = df[0]
        pos_minus[:,:,idx] = df[2]
        pos_plus[:,:,idx] = df[3]
        orientation[:,:,idx] = df[4]
        print('############ Reading Sylinder File = {0}/{1} ({2:.0f}%) ############'.format(
            1+idx,
            len(file_list), 
            100*(1+idx)/len(file_list)), end='\r',flush=True)

    return pos_minus, pos_plus, orientation, gid

def read_all_protein(file_list):

    # get protein number
    numP = count_proteins(file_list[0])

    pos_start = np.zeros( (3,numP, len(file_list)))
    pos_end = np.zeros( (3,numP, len(file_list)))
    gid = np.zeros( (numP, len(file_list)))
    link0 = np.zeros( (numP, len(file_list)))
    link1 = np.zeros( (numP, len(file_list)))
    for idx,fil in enumerate(file_list):
        df = read_dat_protein(fil)
        gid[:,idx] = df[0]
        pos_start[:,:,idx] = df[1]
        pos_end[:,:,idx] = df[2]
        link0[:,idx] = df[3]
        link1[:,idx] = df[4]
        print('############ Reading Protein File = {0}/{1} ({2:.0f}%) ############'.format(
            1+idx,
            len(file_list), 
            100*(1+idx)/len(file_list)), end='\r',flush=True)

    return gid,pos_start,pos_end,link0,link1

def read_dat_sylinder(fname):
    # Read a SylinderAscii_X.dat file

    # open the file and read the lines
    with open(fname, 'r') as file1:
        filecontent = file1.readlines()

        # Delete the first two lines because they dont have any data
        filecontent[0:2] = []

        # Initialize numpy arrays for data
        gid = np.zeros(len(filecontent), dtype=int)
        radius = np.zeros(len(filecontent))
        pos_minus = np.zeros((3, len(filecontent)))
        pos_plus = np.zeros((3, len(filecontent)))
        orientation = np.zeros((3, len(filecontent)))

        for idx, line in enumerate(filecontent):

            # Split the string with space-delimited and convert strings into
            # useful data types
            data = line.split()
            gid[idx] = int(data[1])

            dat = np.array(list(map(float, data[2::])))
            radius[idx] = dat[0]
            pos_minus[:,idx] = dat[1:4]
            pos_plus[:,idx] = dat[4:7]
            xi = pos_plus[:,idx] - pos_minus[:,idx]
            orientation[:,idx] = xi / np.sqrt(xi.dot(xi))
    return gid, radius, pos_minus, pos_plus, orientation

def read_dat_protein(fname):
    # Read a ProteinAscii_X.dat file

    # open the file and read the lines
    with open(fname, 'r') as file1:
        filecontent = file1.readlines()

        # Delete the first two lines because they dont have any data
        filecontent[0:2] = []

        # Initialize numpy arrays for data
        gid = np.zeros(len(filecontent), dtype=int)
        pos0 = np.zeros((3, len(filecontent)))
        pos1 = np.zeros((3, len(filecontent)))
        link0 = np.zeros(len(filecontent), dtype=int)
        link1 = np.zeros(len(filecontent), dtype=int)

        for idx, line in enumerate(filecontent):

            # Split the string with space-delimited and convert strings into
            # useful data types
            data = line.split()
            # pdb.set_trace()
            gid[idx] = int(data[1])
            link0[idx] = int(data[9])
            link1[idx] = int(data[10])
            dat = np.array(list(map(float, data[2:9])))
            pos0[:,idx] = dat[1:4]
            pos1[:,idx] = dat[4::]
    return gid, pos0, pos1, link0, link1


def count_sylinders(fname):
    # count sylinders in a file
    with open(fname, 'r') as file1:
        filecontent = file1.readlines()
    return len(filecontent)-2


def count_proteins(fname):
    # count proteins in a file
    with open(fname, 'r') as file1:
        filecontent = file1.readlines()
    return len(filecontent)-2

def read_all_constraint(file_list):

    Stress_Uni = np.zeros( (2, len(file_list)))
    Stress_Bi = np.zeros( (2, len(file_list)))
    for idx,fil in enumerate(file_list):
        bilat,stress = read_pvtp_constraint(fil)
        stress_uni = []
        stress_bi = []
        for bil,sts in zip(bilat, stress):
            if bil == 0:
                stress_uni.append( sts)
            elif bil == 1:
                stress_bi.append( sts) 
        Stress_Uni[:,idx] = [np.mean(stress_uni), np.std(stress_uni)]
        Stress_Bi[:,idx] = [np.mean(stress_bi), np.std(stress_bi)]
        pdb.set_trace()
    return Stress_Uni, Stress_Bi

def read_pvtp_constraint(fname):
    # Read a ConBlock_X.pvtp file

    # create vtk reader
    reader = vtk.vtkXMLPPolyDataReader()
    reader.SetFileName( str(fname) )
    reader.Update()
    data = reader.GetOutput()

    print('Reading {0}'.format(fname.stem) )
    # fill data
    # step 1, end coordinates
    nObj = int(data.GetPoints().GetNumberOfPoints() / 2)

    Stress = np.zeros(nObj)
    bilateral = np.zeros(nObj)
    data_index = {
        'bilateral' : 1,
        'delta0' : 2,
        'gamma' : 3,
        'kappa' : 4,
        'oneSide' : 0,
        'Stress': 5
        }
    
    for j in range(nObj):
        bilateral[j] = data.GetCellData().GetArray(data_index['bilateral']).GetTuple(j)[0]
        Stress[j] = np.mean( np.array( data.GetCellData().GetArray(data_index['Stress']).GetTuple(j)[0::3] ) )

    return bilateral, Stress

