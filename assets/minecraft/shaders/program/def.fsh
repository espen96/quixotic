#version 150

uniform vec2 OutSize;
uniform vec2 ScreenSize;
uniform float Time;

in vec2 texCoord;
in vec4 skycol;

uniform sampler2D DiffuseSampler;
uniform sampler2D prevsky;
in float skyIntensity;
in vec3 nsunColor;
in float skyIntensityNight;
in float rainStrength;
in float sunIntensity;
in float moonIntensity;


flat in vec3 ambientUp;
flat in vec3 ambientLeft;
flat in vec3 ambientRight;
flat in vec3 ambientB;
flat in vec3 ambientF;
flat in vec3 ambientDown;
flat in vec3 avgSky;
out vec4 fragColor;


#define PI 3.141592


#define SKY_BRIGHTNESS_DAY 0.4//[0.0 0.5 0.75 1.0 1.2 1.4 1.6 1.8 2.0]
#define SKY_BRIGHTNESS_NIGHT 2.1 //[0.0 0.5 0.75 1.0 1.2 1.4 1.6 1.8 2.0]
#define fsign(a)  (clamp((a)*1e35,0.,1.)*2.-1.)


float facos(float inX) {

	const float C0 = 1.56467;
	const float C1 = -0.155972;

    float x = abs(inX);
    float res = C1 * x + C0;
    res *= sqrt(1.0f - x);

    return (inX >= 0) ? res : PI - res;
}

vec3 ScreenSpaceDither( vec2 vScreenPos )
{
    vec3 vDither = vec3( dot( vec2( 171.0, 231.0 ), vScreenPos.xy) );
    vDither.rgb = fract( vDither.rgb / vec3( 103.0, 71.0, 97.0 ) );

    return vDither.rgb / 255.0; //note: looks better without 0.375...
}


vec3 lumaBasedReinhardToneMapping(vec3 color)
{
	float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
	float toneMappedLuma = luma / (1. + luma);
	color *= toneMappedLuma / luma;
	color = pow(color, vec3(1. / 2.2));
	return color;
}


#define MIN_LIGHT_AMOUNT 0.225 //[0.0 0.5 1.0 1.5 2.0 3.0 4.0 5.0]
#define TORCH_AMOUNT 1.0 //[0.0 0.5 0.75 1. 1.2 1.4 1.6 1.8 2.0]
#define TORCH_R 1.0 // [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0]
#define TORCH_G 0.6 // [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0]
#define TORCH_B 0.3 // [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0]

vec3 reinhard_extended(vec3 v, float max_white)
{
    vec3 numerator = v * (1.0f + (v / vec3(max_white * max_white)));
    return numerator / (1.0f + v);
}
vec3 reinhard(vec3 v)
{
    return v / (1.0f + v);
}
float luminance(vec3 v)
{
    return dot(v, vec3(0.2126f, 0.7152f, 0.0722f));
}

vec3 change_luminance(vec3 c_in, float l_out)
{
    float l_in = luminance(c_in);
    return c_in * (l_out / l_in);
}
vec3 reinhard_extended_luminance(vec3 v, float max_white_l)
{
    float l_old = luminance(v);
    float numerator = l_old * (1.0f + (l_old / (max_white_l * max_white_l)));
    float l_new = numerator / (1.0f + l_old);
    return change_luminance(v, l_new);
}
vec3 reinhard_jodie(vec3 v)
{
    float l = luminance(v);
    vec3 tv = v / (1.0f + v);
    return mix(v / (1.0f + l), tv, tv);
}
void main() {
//    vec3 rnd = ScreenSpaceDither( gl_FragCoord.xy );
		vec3 ambientUp = vec3(1.0);

if (gl_FragCoord.x < 17. && gl_FragCoord.y < 17.){


  //luminance (cie model)
	vec3 daySky = vec3(0.0);
	vec3 moonSky = vec3(0.0);
	// Day
	if (skyIntensity > 0.00001)
	{
		vec3 skyColor0 = mix(vec3(0.05,0.5,1.)/1.5,vec3(0.4,0.5,0.6)/1.5,rainStrength*2);
		vec3 skyColor = mix(skyColor0,nsunColor,0.5);
		daySky = skyIntensity*skyColor*vec3(0.8,0.9,1.)*15.*1.5;
	}
	// Night
	if (skyIntensityNight > 0.00001)
	{
		moonSky = skyIntensityNight*vec3(0.08,0.12,0.18)*vec3(0.4)*0.4;
	}


  float skyLut = floor(gl_FragCoord.y)/15.;
  float sky_lightmap = pow(skyLut,2.23);
  float torchLut = floor(gl_FragCoord.x)/15.;
  torchLut *= torchLut;
  float torch_lightmap = ((torchLut*torchLut)*(torchLut*torchLut))*(torchLut*10.)+torchLut;
	float avgEyeIntensity = ((sunIntensity*120.+moonIntensity*4.)+skyIntensity*230.+skyIntensityNight*4.)*sky_lightmap;
	float exposure =  0.18/log2(max(avgEyeIntensity*0.16+1.0,1.13));
  vec3 ambient = (((2.2*(daySky + moonSky))/2.2)*sky_lightmap*log2(1.13+sky_lightmap*1.5)+torch_lightmap*0.05*vec3(TORCH_R,TORCH_G,TORCH_B)*TORCH_AMOUNT)*exposure * vec3(1.0,0.96,0.96)+MIN_LIGHT_AMOUNT*0.001*vec3(0.75,1.0,1.25);
  fragColor = vec4(reinhard_jodie(ambient*10.),1.0);
}

if (gl_FragCoord.x > 8. && gl_FragCoord.x < 9.  && gl_FragCoord.y > 19.+18. && gl_FragCoord.y < 19.+18.+1 ){




  //luminance (cie model)
	vec3 daySky = vec3(0.0);
	vec3 moonSky = vec3(0.0);
	// Day
	if (skyIntensity > 0.00001)
	{
		vec3 skyColor = nsunColor;
		daySky = skyIntensity*skyColor*vec3(0.8,0.9,1.)*15.*1.0;
	}
	// Night
	if (skyIntensityNight > 0.00001)
	{
		moonSky = (skyIntensityNight*vec3(0.08,0.12,0.18)*vec3(0.4))*SKY_BRIGHTNESS_NIGHT;
	}
  fragColor.rgb =  ((daySky + moonSky) )*(1.0-rainStrength);


}




if (gl_FragCoord.x > 18. && gl_FragCoord.y > 1.){
  float cosY = clamp(floor(gl_FragCoord.x - 18.0)/256.*2.0-1.0,-1.0+1e-5,1.0-1e-5);
  cosY = pow(abs(cosY),1/3.0)*fsign(cosY);
  float mCosT = clamp(floor(gl_FragCoord.y-1.0)/256.,0.0,1.0);
  float Y = facos(cosY);
  const float a = -0.8;
  const float b = -0.1;
  const float c = 3.0;
  const float d = -7.;
  const float e = 0.35;

  //luminance (cie model)
	vec3 daySky = vec3(0.0);
	vec3 moonSky = vec3(0.0);
	// Day
	if (skyIntensity > 0.00001)
	{
	  float L0 = (1.0+a*exp(b/mCosT))*(1.0+c*(exp(d*Y)-exp(d*3.1415/2.))+e*cosY*cosY);
		vec3 skyColor0 = mix(vec3(0.05,0.5,1.)/1.5,vec3(0.4,0.5,0.6)/1.5,rainStrength*2);
		vec3 normalizedSunColor = nsunColor;

		vec3 skyColor = mix(skyColor0,normalizedSunColor,1.0-pow(1.0+L0,-1.2))*(1.0-rainStrength);
		daySky = pow(L0,1.0-rainStrength)*skyIntensity*skyColor*vec3(0.8,0.9,1.)*15.*SKY_BRIGHTNESS_DAY;
	}
	// Night
	if (skyIntensityNight > 0.00001)
	{
		float L0Moon = (1.0+a*exp(b/mCosT))*(1.0+c*(exp(d*(PI-Y))-exp(d*3.1415/2.))+e*cosY*cosY);
		moonSky = pow(L0Moon,1.0-rainStrength)*skyIntensityNight*vec3(0.08,0.12,0.18)*vec3(0.4)*SKY_BRIGHTNESS_NIGHT;
	}
  fragColor.rgb = ( ((daySky + moonSky) ) );
}


}
