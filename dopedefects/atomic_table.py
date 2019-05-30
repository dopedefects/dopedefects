"""
Periodic Table
"""
def atomic_weight(x):
    """
    A map containing the elements used in the project with their charge
    and weight.
    Inputs
    ------
    x:    Atomic symbol as string

    Outputs
    -------
    tuple containing the atomic number and the average atomic mass
    """
    return{
        'H':	(1, 1.0079),
        'He':	(2, 4.0026),
        'Li':	(3, 6.941),
        'Be':	(4, 9.0122),
        'B':	(5, 10.811),
        'C':	(6, 12.0107),
        'N':	(7, 14.0067),
        'O':	(8, 15.9994),
        'F':	(9, 18.9984),
        'Ne':	(10, 20.1797),
        'Na':	(11, 22.9897),
        'Mg':	(12, 24.305),
        'Al':	(13, 26.9815),
        'Si':	(14, 28.0855),
        'P':	(15, 30.9738),
        'S':	(16, 32.065),
        'Cl':	(17, 35.453),
        'K':	(19, 39.0983),
        'Ar':	(18, 39.948),
        'Ca':	(20, 40.078),
        'Sc':	(21, 44.9559),
        'Ti':	(22, 47.867),
        'V':	(23, 50.9415),
        'Cr':	(24, 51.9961),
        'Mn':	(25, 54.938),
        'Fe':	(26, 55.845),
        'Ni':	(28, 58.6934),
        'Co':	(27, 58.9332),
        'Cu':	(29, 63.546),
        'Zn':	(30, 65.39),
        'Ga':	(31, 69.723),
        'Ge':	(32, 72.64),
        'As':	(33, 74.9216),
        'Se':	(34, 78.96),
        'Br':	(35, 79.904),
        'Kr':	(36, 83.8),
        'Rb':	(37, 85.4678),
        'Sr':	(38, 87.62),
        'Y':	(39, 88.9059),
        'Zr':	(40, 91.224),
        'Nb':	(41, 92.9064),
        'Mo':	(42, 95.94),
        'Tc':	(43, 98),
        'Ru':	(44, 101.07),
        'Rh':	(45, 102.9055),
        'Pd':	(46, 106.42),
        'Ag':	(47, 107.8682),
        'Cd':	(48, 112.818),
        'In':	(49, 114.818),
        'Sn':	(50, 118.71),
        'Sb':	(51, 121.76),
        'I':	(53, 126.9045),
        'Te':	(52, 127.6),
        'Xe':	(54, 131.293),
        'Cs':	(55, 132.9055),
        'Ba':	(56, 137.327),
        'La':	(57, 138.9055),
        'Ce':	(58, 140.116),
        'Pr':	(59, 140.9077),
        'Nd':	(60, 144.24),
        'Pm':	(61, 145),
        'Sm':	(62, 150.36),
        'Eu':	(63, 151.964),
        'Gd':	(64, 157.25),
        'Tb':	(65, 158.9253),
        'Dy':	(66, 162.5),
        'Ho':	(67, 164.9303),
        'Er':	(68, 167.259),
        'Tm':	(69, 168.9342),
        'Yb':	(70, 173.04),
        'Lu':	(71, 174.967),
        'Hf':	(72, 178.49),
        'Ta':	(73, 180.9479),
        'W':	(74, 183.84),
        'Re':	(75, 186.207),
        'Os':	(76, 190.23),
        'Ir':	(77, 192.217),
        'Pt':	(78, 195.078),
        'Au':	(79, 196.9665),
        'Hg':	(80, 200.59),
        'Tl':	(81, 204.3833),
        'Pb':	(82, 207.2),
        'Bi':	(83, 208.9804),
        'Po':	(84, 209),
        'At':	(85, 210),
        'Rn':	(86, 222),
        'Fr':	(87, 223),
        'Ra':	(88, 226),
        'Ac':	(89, 227),
        'Pa':	(91, 231.0359),
        'Th':	(90, 232.0391),
        'Np':	(93, 237),
        'U':	(92, 238.0289),
        'Am':	(95, 243),
        'Pu':	(94, 244),
        'Cm':	(96, 247),
        'Bk':	(97, 247),
        'Cf':	(98, 251),
        'Es':	(99, 252),
        'Fm':	(100, 257),
        'Md':	(101, 258),
        'No':	(102, 259),
        'Rf':	(104, 261),
        'Lr':	(103, 262),
        'Db':	(105, 262),
        'Bh':	(107, 264),
        'Sg':	(106, 266),
        'Mt':	(109, 268),
        'Rg':	(111, 272),
        'Hs':	(108, 277),
        'Ds':	(110, 0),
        'Cn':	(112, 0),
        'Nh':	(113, 0),
        'Fl':	(114, 0),
        'Mc':	(115, 0),
        'Lv':	(116, 0),
        'Ts':	(117, 0),
        'Og':	(118, 0)
        }.get(x, 'H')
