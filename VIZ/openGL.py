### examples

void main()
{
    gl_Position = vec4(0.0,0.0,0.0,1.0);
}

# fragment shader

void main()
{
    gl_FragColor = vec4(0.0,0.0,0.0,1.0);
}

data = numpy.zeros(4, dtype = [ ("position", np.float32, 3),
                                ("color",    np.float32, 4)] )
                                
attribute vec2 position;
attribute vec4 color;
void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
}
