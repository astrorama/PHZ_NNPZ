from nnpz.io.output_column_providers.McPdf1DMedian import McPdf1DMedian

from .fixtures import *


def test_pdf_median(sampler: McSampler, mock_output_handler: MockOutputHandler,
                       contributions: np.ndarray):
    contributions_over =  np.array(
        [(0, [0, 1, 2], [1., 0., 0.]), (1, [0, 1, 2], [0., 0., 1.])],
        dtype=[('ID', int), ('NEIGHBOR_INDEX', np.float32, 3), ('NEIGHBOR_WEIGHTS', np.float32, 3)])

    pdfmedian = McPdf1DMedian(sampler, param_name='M1')
    mock_output_handler.add_column_provider(sampler)
    mock_output_handler.add_column_provider(pdfmedian)
    mock_output_handler.initialize(nrows=len(contributions_over))
    mock_output_handler.write_output_for(np.arange(len(contributions_over)), contributions_over)
    columns = mock_output_handler.get_data_for_provider(pdfmedian)

    assert len(columns.dtype.fields) == 1
    assert 'PHZ_PP_MEDIAN_M1' in columns.dtype.fields
    column = columns['PHZ_PP_MEDIAN_M1']
    assert column.shape == (2,)
    assert column[0] >10  # from samples for peak=4 
    assert np.abs(column[1] -10)<3  # gaussian centered in 10 with sigma=1

