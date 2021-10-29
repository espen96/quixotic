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
noperspective in vec3 pos1;
noperspective in vec3 pos2;
in vec3 gtime;
noperspective in vec3 pos3;
in vec2 texCoord0;
in vec2 texCoord1;
in vec4 normal;
in vec4 glpos;

out vec4 fragColor;

float luma(vec3 color) {
    return dot(color, vec3(0.299, 0.587, 0.114));
}

float hash(float n) {
    return fract(tan(n) * 43758.5453);
}

float getNoise(vec3 x) {
    x *= 50.0;
    // The noise function returns a value in the range -1.0f -> 1.0f

    vec3 p = floor(x);
    vec3 f = fract(x);

    f = f * f * (3.0 - 2.0 * f);
    float n = p.x + p.y * 57.0 + 113.0 * p.z;

    return mix(mix(mix(hash(n + 0.0), hash(n + 1.0), f.x), mix(hash(n + 57.0), hash(n + 58.0), f.x), f.y), mix(mix(hash(n + 113.0), hash(n + 114.0), f.x), mix(hash(n + 170.0), hash(n + 171.0), f.x), f.y), f.z);
}

float getLayeredNoise(vec3 seed) {
    return (0.5 * getNoise(seed * 0.05)) +
        (0.25 * getNoise(seed * 0.1)) +
        (0.125 * getNoise(seed * 0.2)) +
        (0.0625 * getNoise(seed * 0.4));
}
float cubeSmooth(float x) {
    return (x * x) * (3.0 - 2.0 * x);
}

float TextureCubic(sampler2D tex, vec2 pos) {
    ivec2 texSize = textureSize(tex, 0) * 5;
    vec2 texelSize = (1.0 / vec2(texSize));
    float p0q0 = texture(tex, pos).a;
    float p1q0 = texture(tex, pos + vec2(texelSize.x, 0)).a;

    float p0q1 = texture(tex, pos + vec2(0, texelSize.y)).a;
    float p1q1 = texture(tex, pos + vec2(texelSize.x, texelSize.y)).a;

    float a = cubeSmooth(fract(pos.x * texSize.x));

    float pInterp_q0 = mix(p0q0, p1q0, a);
    float pInterp_q1 = mix(p0q1, p1q1, a);

    float b = cubeSmooth(fract(pos.y * texSize.y));

    return mix(pInterp_q0, pInterp_q1, b);
}

void main() {
    float aspectRatio = ScreenSize.x / ScreenSize.y;

    if(gl_PrimitiveID >= 2) {

        gl_FragDepth = 0.0;
    }

    fragColor = texture(Sampler0, (vec2((gl_FragCoord.xy / ScreenSize) / vec2(1, aspectRatio))));    
    
//    fragColor.a =  0.1;
        int index = inControl(gl_FragCoord.xy, ScreenSize.x);
    // currently in a control/message pixel
    if(index != -1) {
        vec3 position = vec3(0.0);

        //if (index == 50 ) fragColor = vec4( encodeFloat24( pos.y-125 ),1);
        if (index == 50 ) fragColor = vec4( pos1,1);
        if (index == 51 ) fragColor = vec4( pos2,1);
        if (index == 53 ) fragColor = vec4( pos3,1);
        if (index == 54 ) fragColor = vec4( gtime,1);

    }


//    fragColor.a *= (luma(FogColor.rgb)*luma(FogColor.rgb));
 //   fragColor.a *=  1-(fogValue*0.25);
}
