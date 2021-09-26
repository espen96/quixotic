#version 150

uniform sampler2D DiffuseSampler;
uniform sampler2D DiffuseDepthSampler;
uniform vec2 OutSize;
uniform vec2 ScreenSize;
uniform float Time;

in vec2 texCoord;
in vec2 oneTexel;



in vec4 skycol;
in float aspectRatio;

in float skyIntensity;
in vec3 nsunColor;
in float skyIntensityNight;
in float rainStrength;



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






void main() {
//    vec3 rnd = ScreenSpaceDither( gl_FragCoord.xy );

    float aspectRatio = ScreenSize.x/ScreenSize.y;
        vec4 screenPos = gl_FragCoord;
        screenPos.xy = (screenPos.xy / ScreenSize - vec2(0.5)) * 2.0;
        screenPos.zw = vec2(1.0);


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
