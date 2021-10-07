#version 150

#moj_import <fog.glsl>
#moj_import <utils.glsl>
#moj_import <mappings.glsl>

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
noperspective in vec3 test;
in vec2 texCoord0;
in vec2 texCoord2;
in vec2 texCoord3;
in vec4 normal;
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
#define Bayer512(a) (Bayer256(0.5 * (a)) * 0.25 + Bayer2(a))
#define Bayer1024(a) (Bayer512(0.5 * (a)) * 0.25 + Bayer2(a))

float map(float value, float min1, float max1, float min2, float max2) {
  return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
}


void main() {

  vec3 rnd = ScreenSpaceDither( gl_FragCoord.xy );


  discardControlGLPos(gl_FragCoord.xy, glpos);
                  

  float alpha = textureLod(Sampler0, texCoord0,0).a; 
  vec4 albedo = textureLod(Sampler0, texCoord0,0) ;
  vec4 color = texture(Sampler0, texCoord0,0) * vertexColor * ColorModulator;
  if(alpha <0.15)color = textureLod(Sampler0, texCoord0,0) * vertexColor * ColorModulator;
  color.a = alpha;
  float lightm = 0;


//  color.rgb = (color.rgb*lm2.rgb);
        

  color.rgb +=rnd/16; 
    color.rgb = clamp(color.rgb,0.01,1);
    #define sssMin 22
    #define sssMax 47
    #define lightMin 48
    #define lightMax 72
    #define roughMin 73
    #define roughMax 157
    #define metalMin 158
    #define metalMax 251

  float translucent = 0;

  float mod2 = gl_FragCoord.x + gl_FragCoord.y;
  float res = mod(mod2, 2.0f);

    float lum = luma4(albedo.rgb);
	vec3 diff = albedo.rgb-lum;

   float alpha0 = int(textureLod(Sampler0, texCoord0,0).a*255);
  float procedual1 = ((distance(textureLod(Sampler0, texCoord0,0).rgb,test.rgb)))*255;
 //if (alpha0 ==255) {alpha0 = map(procedual1,0,255,roughMin,roughMax-16);
 //}

//color.rgb = test;
if(alpha0==255){
vec3 test2 = floor(test.rgb*255);
float test3  = floor(test2.r+test2.g+test2.b);

 if(test3 <= 560 && test3 >= 550)  alpha0 = clamp(procedual1*albedo.r,lightMin,lightMax);
 if(test3 == 382 && test2.b == 83)  alpha0 = clamp((color.r*color.r*color.r)*255,lightMin,lightMax);
 if(test3 <= 316 && test3 >= 310)  alpha0 = clamp(procedual1,lightMax,lightMax);
}
// if(test3 <= 316 && test3 >= 310)  color.rgb = vec3(1,0,0);
//    if(alpha0 >=  sssMin && alpha0 <=  sssMax)   alpha0 = clamp(alpha0+noise,sssMin,sssMax); // SSS

  float noise = luma4(rnd)*255;  
 
    if(alpha0 >=  sssMin && alpha0 <=  sssMax)   alpha0 = int(clamp(alpha0+0,sssMin,sssMax)); // SSS

    if(alpha0 >=  lightMin && alpha0 <= lightMax)   alpha0 = int(clamp(alpha0+noise,lightMin,lightMax)); // Emissives

    if(alpha0 >= roughMin && alpha0 <= roughMax)   alpha0 = int(clamp(alpha0+noise,roughMin,roughMax)); // Roughness


    if(alpha0 >= metalMin && alpha0 <= metalMax)   alpha0 = int(clamp(alpha0+noise,metalMin,metalMax)); // Metals

  noise /= 255;  


  float alpha1 = 0.0;
  float alpha2 = 0.0;

  if(alpha0 <= 128) alpha1 = floor(map( alpha0,  0, 128, 0, 255))/255;
  if(alpha0 >= 128) alpha2 = floor(map( alpha0,  128, 255, 0, 255))/255;

  //  fragColor = linear_fog(color, vertexDistance, FogStart, FogEnd, FogColor);

  float alpha3 = alpha1;
  float lm = lmx;
  if (res == 0.0f)    {
    lm = lmy;
    alpha3 = alpha2;
    //color.rgb = normal.rgb;
  }
  fragColor = color;
//  fragColor.rgb = test.rgb;
//if(int(textureLod(Sampler0, texCoord0,0).a*255)==255)alpha3 = 1.0;   
  fragColor.a = packUnorm2x4( alpha3,clamp(lm+(luma4(rnd*clamp(lm*100,0,1))/2),0,0.95));
  
  
  }
