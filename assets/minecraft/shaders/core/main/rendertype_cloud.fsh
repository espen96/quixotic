#version 150

#moj_import <fog.glsl>
#moj_import <utils.glsl>

uniform sampler2D Sampler0;
uniform sampler2D Sampler1;

uniform vec4 ColorModulator;
uniform vec2 ScreenSize;
uniform float FogStart;
uniform float FogEnd;
uniform vec4 FogColor;

in float vertexDistance;
in vec4 vertexColor;
in vec3 pos;
in vec2 texCoord0;
in vec2 texCoord1;
in vec4 normal;
in vec4 glpos;

out vec4 fragColor;

float luma(vec3 color){
	return dot(color,vec3(0.299, 0.587, 0.114));
}


float hash( float n )
{
    return fract(tan(n)*43758.5453);
}

float getNoise( vec3 x )
{
    x *= 50.0;
    // The noise function returns a value in the range -1.0f -> 1.0f

    vec3 p = floor(x);
    vec3 f = fract(x);

    f       = f*f*(3.0-2.0*f);
    float n = p.x + p.y*57.0 + 113.0*p.z;

    return mix(mix(mix( hash(n+0.0), hash(n+1.0),f.x),
                   mix( hash(n+57.0), hash(n+58.0),f.x),f.y),
               mix(mix( hash(n+113.0), hash(n+114.0),f.x),
                   mix( hash(n+170.0), hash(n+171.0),f.x),f.y),f.z);
}

float getLayeredNoise(vec3 seed)
{
	return (0.5 * getNoise(seed * 0.05)) +
           (0.25 * getNoise(seed * 0.1)) +
           (0.125 * getNoise(seed * 0.2)) +
           (0.0625 * getNoise(seed * 0.4));
}

void main() {
    	float aspectRatio = ScreenSize.x/ScreenSize.y;

    if (gl_PrimitiveID >= 2) {

            gl_FragDepth = 0.0;
    }

    fragColor = texture(Sampler0, vec2((gl_FragCoord.xy/256)) );    
//    fragColor.a =  0.1;
        int index = inControl(gl_FragCoord.xy, ScreenSize.x);
    // currently in a control/message pixel
    if(index != -1) {
        vec3 position = vec3(0.0);
        if(pos.y > 0) position.x = abs(pos.y)/128;    
        if(pos.y < 0) position.z = abs(pos.y)/128;    
        if (index == 50 ) fragColor = vec4( encodeFloat24(pos.y-125),1);


    }



//    fragColor.a *= (luma(FogColor.rgb)*luma(FogColor.rgb));
 //   fragColor.a *=  1-(fogValue*0.25);
}
