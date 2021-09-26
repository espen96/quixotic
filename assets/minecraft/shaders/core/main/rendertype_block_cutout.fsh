#version 150

#moj_import <fog.glsl>
#moj_import <utils.glsl>

uniform sampler2D Sampler0;
uniform sampler2D Sampler2;
uniform vec2 ScreenSize;

uniform vec4 ColorModulator;
uniform float FogStart;
uniform float FogEnd;
uniform vec4 FogColor;
in vec3 chunkOffset;

in float vertexDistance;
in float lm;
in vec4 vertexColor;
in vec4 lm2;
in vec2 texCoord0;
in vec2 texCoord2;
in vec2 texCoord3;
in vec4 normal;
in vec4 test;
in vec4 glpos;
in float lmx;
in float lmy;
out vec4 fragColor;

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

float map(float value, float min1, float max1, float min2, float max2) {
  return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
}
void main() {

    vec3 rnd = ScreenSpaceDither( gl_FragCoord.xy );


        discardControlGLPos(gl_FragCoord.xy, glpos);
                  
    vec4 albedo =textureLod(Sampler0, texCoord0,0);

    vec4 color = texture(Sampler0, texCoord0) * vertexColor * ColorModulator;
 
    float lightm = 0;


    color.rgb = (color.rgb*lm2.rgb);

    color.a = textureLod(Sampler0, texCoord0,0).a;         
    if (color.a*255 <= 17.0) {
        discard;
    }
 //   color.rgb += FogColor.rgb*0.1;
    color.rgb +=rnd/255; 



    float lum = luma4(albedo.rgb);
	vec3 diff = albedo.rgb-lum;

    float translucent = 0;

    float mod2 = gl_FragCoord.x + gl_FragCoord.y;
    float res = mod(mod2, 2.0f);

  float alpha0 = int(textureLod(Sampler0, texCoord0,0).a*255);
  float alpha1 = 0.0;
  float alpha2 = 0.0;
	float lAlbedoP = length(albedo);




//	alpha0 = map(lAlbedoP*255,0,255,116,208);
if(alpha0 <= 128) alpha1 = floor(map( alpha0,  0, 128, 0, 255))/255;
if(alpha0 >= 128) alpha2 = floor(map( alpha0,  128, 255, 0, 255))/255;


//     fragColor = linear_fog(color, vertexDistance, FogStart, FogEnd, FogColor);

    float alpha3 = alpha1;
    if (diff.r < 0.1) translucent = albedo.g;
   float lm = lmx;
  if (res == 0.0f)    {
    lm = lmy;
    translucent = lightm;
     alpha3 = alpha2;
//         color.rgb = normal.rgb;
  }
 
    fragColor = color;
   
    fragColor.a = packUnorm2x4( alpha3,clamp(lm+(Bayer256(gl_FragCoord.xy)/16),0,0.9));
}
