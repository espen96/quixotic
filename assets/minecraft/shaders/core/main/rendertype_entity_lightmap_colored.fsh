#version 150

#moj_import <fog.glsl>
#moj_import <utils.glsl>

uniform sampler2D Sampler0;
uniform sampler2D Sampler1;

uniform vec4 ColorModulator;
uniform float FogStart;
uniform float FogEnd;
uniform vec4 FogColor;

in float vertexDistance;
in vec4 vertexColor;
in vec2 texCoord0;
in vec2 texCoord1;
in vec4 normal;
in vec4 glpos;
in mat4 ProjMat2;
out vec4 fragColor;
in float lmx;
in float lmy;

float ditherGradNoise() {
  return fract(52.9829189 * fract(0.06711056 * gl_FragCoord.x + 0.00583715 * gl_FragCoord.y));
}

float packUnorm2x4(vec2 xy) {
	return dot(floor(15.0 * xy + 0.5), vec2(1.0 / 255.0, 16.0 / 255.0));
}
float packUnorm2x2(vec2 xy) {
	return dot(floor(4.0 * xy + 0.5), vec2(1.0 / 16.0, 16.0 / 16.0));
}
float packUnorm2x4(float x, float y) { return packUnorm2x4(vec2(x, y)); }
vec2 unpackUnorm2x4(float pack) {
	vec2 xy; xy.x = modf(pack * 255.0 / 16.0, xy.y);
	return xy * vec2(16.0 / 15.0, 1.0 / 15.0);
}
vec2 unpackUnorm2x2(float pack) {
	vec2 xy; xy.x = modf(pack * 255.0 / 16.0, xy.y);
	return xy * vec2(16.0 / 15.0, 1.0 / 15.0);
}
//Dithering from Jodie
float Bayer2(vec2 a) {
    a = floor(a+fract(GameTime * 1200));
    return fract(dot(a, vec2(0.5, a.y * 0.75)));
}

#define Bayer4(a)   (Bayer2(  0.5 * (a)) * 0.25 + Bayer2(a))
#define Bayer8(a)   (Bayer4(  0.5 * (a)) * 0.25 + Bayer2(a))
#define Bayer16(a)  (Bayer8(  0.5 * (a)) * 0.25 + Bayer2(a))
#define Bayer32(a)  (Bayer16( 0.5 * (a)) * 0.25 + Bayer2(a))
#define Bayer64(a)  (Bayer32( 0.5 * (a)) * 0.25 + Bayer2(a))
#define Bayer128(a) (Bayer64( 0.5 * (a)) * 0.25 + Bayer2(a))
#define Bayer256(a) (Bayer128(0.5 * (a)) * 0.25 + Bayer2(a))



void main() {
    discardControlGLPos(gl_FragCoord.xy, glpos);
 //   vec4 color = texture(Sampler0, texCoord0) * vertexColor * ColorModulator;
    vec4 color = texture(Sampler0, texCoord0) * 1 * ColorModulator;

  if (color.a*255 <= 17.0) {
    discard;
  }
//  fragColor = linear_fog(color, vertexDistance, FogStart, FogEnd, FogColor);
    fragColor = color;
    color.rgb = clamp(color.rgb,0.01,1);
    float mod2 = gl_FragCoord.x + gl_FragCoord.y;
    float res = mod(mod2, 2.0f);



float lm = lmx;
  if (res == 0.0f)    {
    lm = lmy;
  }

   
    fragColor.a = packUnorm2x4( 0,clamp(lm+(Bayer256(gl_FragCoord.xy)/16),0,0.9));


}
