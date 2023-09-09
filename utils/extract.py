from os import SEEK_END, SEEK_CUR

bias_indices = ['bias_95_tri', 'bias_95_uran', 'bias_95_al', 'bias_95_ph', 'bias_110_tri', 'bias_110_uran', 'bias_110_ph']

def getBiasFactors(sim_path, para_ens):
    fnum = int( ''.join(list(filter(str.isdigit, sim_path.name))) )
    bias_factors = para_ens.loc[fnum-1, bias_indices]
    return bias_factors

def listNlines(fname, n = 1):
    '''implementation source: https://stackoverflow.com/questions/46258499/how-to-read-the-last-line-of-a-file-in-python'''

    '''Returns the nth before last line of a file as a list of string(s)
       n=1 gives last line, i.e., a list of length 1
    '''
    numl = 0
    with open(fname, 'rb') as f:
        try:
            f.seek(-2, SEEK_END)
            while numl < n:
                f.seek(-2, SEEK_CUR)
                if f.read(1) == b'\n':
                    numl += 1
        except OSError:
            f.seek(0)

        lines = []
        for i in range(n):
            lines.append(f.readline().decode())
    return lines
