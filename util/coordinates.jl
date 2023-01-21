using LinearAlgebra


function polar2cart(z; l=1.0)
    "Map angles to Cartesian space"

    # Position of first mass
    x = l*sin.(z[1])
    y = l*-cos.(z[1])
    
    return (x,y)
end

function polar2cart_com(z; l1=1, l2=1)
    "Map angles of centers of masses to Cartesian space"

    # Position of first mass
    x1 = l1/2*sin(z[1])
    y1 = l1/2*-cos(z[1])
    
    # Position of second mass
    x2 =  l1*sin(z[1]) + l2/2*sin(z[2])
    y2 = -l1*cos(z[1]) - l2/2*cos(z[2])
    
    return (x1,y1), (x2,y2)
end

function polar2cart_rod(z; l1=1, l2=1)
    "Map end points of rods to Cartesian space"

    # End position of first rod
    x1 = l1*sin(z[1])
    y1 = l1*-cos(z[1])
    
    # End position of second mass
    x2 = x1 + l2*sin(z[2])
    y2 = y1 - l2*cos(z[2])
    
    return (x1,y1), (x2,y2)
end