import vtk
import numpy as np
from vtk.util.numpy_support import numpy_to_vtk
from pathlib import Path


def add_array_to_vtk(array2D, arrayName, sim_path, data_name='Sylinder'):
   
    print('Writing {0} to vtk files'.format(arrayName))

    def fileKey(f):
        return int( f.parts[-1].split('_')[-1].split('.pvtp')[0] )

    # assemble files to save to
    files = sorted( list(Path(sim_path).glob('**/Sylinder*.pvtp')), key=fileKey)

    # add array to files
    for jt,fil in enumerate(files):
        fil = str(fil)
        
        # Array to write
        data_add = array2D[:,jt].reshape(-1,1)
        data_add = np.asarray(data_add)
        data_add = np.atleast_2d(data_add)
        array_add = numpy_to_vtk(data_add, deep=True)
        array_add.SetName(arrayName)

        reader = vtk.vtkXMLPPolyDataReader()
        reader.SetFileName(fil)
        reader.Update()
        data = reader.GetOutput()

        # Add Test Array
        data.GetCellData().AddArray(array_add)
        data.GetCellData().Modified()
        reader.Update()
        data = reader.GetOutput()

        # write
        writer = vtk.vtkXMLPPolyDataWriter()
        writer.SetFileName(fil)
        writer.SetInputData(data)
        writer.Write()

if __name__ == '__main__':

    # test
    sim_path = 'Test'
    array2D = np.random.rand(1600,902)
    array2D = np.array( array2D>0.5, dtype='int')
    add_array_to_vtk(array2D, 'TESTARRAY', sim_path, data_name='Sylinder')
