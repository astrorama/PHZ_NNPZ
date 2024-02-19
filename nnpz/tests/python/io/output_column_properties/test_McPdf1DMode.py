from nnpz.io.output_column_providers.McPdf1DMode import McPdf1DMode

from .fixtures import *


def test_pdf_mode(sampler: McSampler, mock_output_handler: MockOutputHandler,
                       contributions: np.ndarray):
    contributions_over =  np.array(
        [(0, [0, 1, 2], [1., 0., 0.]), (1, [0, 1, 2], [0., 0., 1.])],
        dtype=[('ID', int), ('NEIGHBOR_INDEX', np.float32, 3), ('NEIGHBOR_WEIGHTS', np.float32, 3)])

    pdfmode = McPdf1DMode(sampler, param_name='M1')
    mock_output_handler.add_column_provider(sampler)
    mock_output_handler.add_column_provider(pdfmode)
    mock_output_handler.initialize(nrows=len(contributions_over))
    mock_output_handler.write_output_for(np.arange(len(contributions_over)), contributions_over)
    columns = mock_output_handler.get_data_for_provider(pdfmode)

    assert len(columns.dtype.fields) == 1
    assert 'PHZ_PP_MODE_M1' in columns.dtype.fields
    column = columns['PHZ_PP_MODE_M1']
    assert column.shape == (2,)
    assert np.abs(column[0] - 4)<2
    assert np.abs(column[1] - 10)<3 

