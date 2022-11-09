"""Extension for generic linear program node."""
import struct

from gp.core.math import clog
import gp.core.node as node

class Node(node.Node):
    """Class extension for generic linear program node."""

    def machine_code(self, w_opcode, w_depth, w_value, form='x'):
        """Return machine code for `node.Node` object."""
        if form not in 'bdxX':
            # Invalid format specification.
            raise ValueError(f'Value provided for argument `form`, '
                             f'`{form}`, is invalid.')
        
        # Machine code prefix.
        prefix = f'0{form}' if form != 'd' else ''

        # Exctract the single-precision IEEE-754 encoded value of 
        # `self.value`, and convert the encoded value into an integer.
        value = struct.unpack("<I", struct.pack("<f", self.value))[0]

        # Calculate the number of digits needed to represent
        # the machine code in the relevant "form".
        if form == 'b':
            base = 2
        elif form == 'd':
            base = 10
        else:
            base = 16
        w_opcode = int(clog(2 ** w_opcode, base))
        w_depth = int(clog(2 ** w_depth, base))
        w_value = int(clog(2 ** w_value, base))

        # Return machine code.
        return(f'{prefix}' 
                + f'{self.opcode:0{w_opcode}{form}}'[-w_opcode:] 
                + f'{self.depth:0{w_depth}{form}}'[-w_depth:]
                + f'{value:0{w_value}{form}}'[:w_value])