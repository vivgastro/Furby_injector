import numpy as np

def get_telescope_params(f):
    fbottom = f.header.fch1 - f.header.foff / 2
    ftop = f.header.fch1 - f.header.foff / 2 + f.header.foff * f.header.nchans

    if f.header.foff < 0:
        ftop, fbottom = fbottom, ftop

    params = {
            'name': f.header.telescope,
            'nch': f.header.nchans,
            'ftop':ftop, 
            'fbottom': fbottom, 
            'tsamp': f.header.tsamp
            }

    return params

def main(args):
    filname = args.fil
    f = F(filname)
    params = get_telescope_params(f)
    print(f"The telescope params are: {params}")

    if args.save:
        print("Saving the params to ", args.save)
        with open(args.save, 'w') as o:
            for key, val in params.items():
                o.write(str(key)+ ": "+ str(val)+ "\n")


if __name__ == '__main__':
    from sigpyproc.readers import FilReader as F
    import argparse

    a = argparse.ArgumentParser()
    a.add_argument("fil", type=str, help="Path to the filterbank from which we need to read the params")
    a.add_argument("-save", type=str, help="Save the params to a file with the given name")
    args = a.parse_args()
    main(args)



