from nnpz.io.output_column_providers.McPdf1DQuantile import McPdf1DQuantile

from .fixtures import *


def test_pdf_mode(sampler: McSampler, mock_output_handler: MockOutputHandler,
                       contributions: np.ndarray):
    contributions_over =  np.array(
        [(0, [0, 1, 2], [1., 0., 0.]), (1, [0, 1, 2], [0., 0., 1.])],
        dtype=[('ID', int), ('NEIGHBOR_INDEX', np.float32, 3), ('NEIGHBOR_WEIGHTS', np.float32, 3)])

    pdfquantile = McPdf1DQuantile(sampler, param_name='M1', range_pcent=68)
    mock_output_handler.add_column_provider(sampler)
    mock_output_handler.add_column_provider(pdfquantile)
    mock_output_handler.initialize(nrows=len(contributions_over))
    mock_output_handler.write_output_for(np.arange(len(contributions_over)), contributions_over)
    columns = mock_output_handler.get_data_for_provider(pdfquantile)

    assert len(columns.dtype.fields) == 1
    assert 'PHZ_PP_68_M1' in columns.dtype.fields
    column = columns['PHZ_PP_68_M1']
    assert column.shape == (2,2)
    assert column[0][0] < column[0][1]  
    assert column[1][0] < column[1][1]  
    assert np.abs(column[1][0] -9)<2  
    assert np.abs(column[1][1] -11)<2  

