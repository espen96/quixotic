#version 150

uniform sampler2D temporals3Sampler;
uniform sampler2D noisetex;
uniform sampler2D DiffuseDepthSampler;

uniform vec2 OutSize;
uniform vec2 ScreenSize;
uniform float Time;

in vec2 texCoord;
in vec2 oneTexel;
in vec3 avgSky;

in vec4 fogcol;
in float near;

in vec4 skycol;
in vec4 rain;
in float aspectRatio;
in mat4 gbufferModelViewInverse;
in float sunElevation;
in float rainStrength;
in vec3 sunVec;

out vec4 fragColor;
#define CLOUDS_QUALITY 0.5 





vec3 toLinear(vec3 sRGB){
	return sRGB * (sRGB * (sRGB * 0.305306011 + 0.682171111) + 0.012522878);
}






////////////////////////////////////////////////

float frameTimeCounter =  sunElevation*1000;

const float PI = 3.141592;
const vec3 cameraPosition = vec3(0);
const float cloud_height = 1500.;
const float maxHeight = 1650.;
int maxIT_clouds = 15;
const float steps = 15.0;
const float cdensity = 0.20;


///////////////////////////




vec3 ScreenSpaceDither(vec2 vScreenPos)
{
    vec3 vDither = vec3(dot(vec2(131.0, 312.0), vScreenPos.xy + fract(Time*2048)));
    vDither.rgb = fract(vDither.rgb / vec3(103.0, 71.0, 97.0)) * vec3(2.0,2.0,2.0) - vec3(0.5, 0.5, 0.5);
    return (vDither.rgb / steps);
}






	//Mie phase function
float phaseg(float x, float g){
    float gg = g * g;
    return (gg * -0.25 /3.14 + 0.25 /3.14) * pow(-2.0 * (g * x) + (gg + 1.0), -1.5);
}											   



float cubeSmooth(float x) {
    return (x * x) * (3.0 - 2.0 * x);
}



float TextureCubic(sampler2D tex, vec2 pos) {
    ivec2 texSize = textureSize(tex, 0) * 5;
    vec2 texelSize = (1.0/vec2(texSize));    
    float p0q0 = texture(tex, pos).a;
    float p1q0 = texture(tex, pos + vec2(texelSize.x, 0)).a;

    float p0q1 = texture(tex, pos + vec2(0, texelSize.y)).a;
    float p1q1 = texture(tex, pos + vec2(texelSize.x , texelSize.y)).a;

    float a = cubeSmooth(fract(pos.x * texSize.x));

    float pInterp_q0 = mix(p0q0, p1q0, a);
    float pInterp_q1 = mix(p0q1, p1q1, a);

    float b = cubeSmooth(fract(pos.y*texSize.y));

    return mix(pInterp_q0, pInterp_q1, b);
}

//Cloud without 3D noise, is used to exit early lighting calculations if there is no cloud
float cloudCov(in vec3 pos,vec3 samplePos){
	float mult = max(pos.y-2000.0,0.0)/2000.0;
	float mult2 = max(-pos.y+2000.0,0.0)/500.0;
	float coverage = clamp(texture(noisetex,(samplePos.xz/12500.)).x-0.2,0.0,1.0)/(0.8);
	float cloud = coverage*coverage*1.0 - mult*mult*mult*3.0 - mult2*mult2;
	return max(cloud, 0.0);
}
//Erode cloud with 3d Perlin-worley noise, actual cloud value

	float cloudVol(in vec3 pos,in vec3 samplePos,in float cov){
		float mult2 = (pos.y-1500)/2500.0;
		float cloud = clamp(cov-0.11*(0.2+mult2),0.0,1.0);
		return cloud;

	}





vec4 renderClouds(vec3 fragpositi, vec3 color,float dither,vec3 sunColor,vec3 moonColor,vec3 avgAmbient) {


		vec4 fragposition = vec4(fragpositi,1.0);

		vec3 worldV = normalize(fragposition.rgb);
		float VdotU = worldV.y;
		maxIT_clouds = int(clamp(maxIT_clouds/sqrt(VdotU),0.0,maxIT_clouds));

		vec3 dV_view = worldV;


		vec3 progress_view = dV_view*dither+cameraPosition;

		float total_extinction = 1.0;


		worldV = normalize(worldV)*300000. + cameraPosition; //makes max cloud distance not dependant of render distance
		if (worldV.y < cloud_height) return vec4(0.,0.,0.,1.);	//don't trace if no intersection is possible


		dV_view = normalize(dV_view);

		//setup ray to start at the start of the cloud plane and end at the end of the cloud plane
		dV_view *= max(maxHeight-cloud_height, 0.0)/dV_view.y/maxIT_clouds;

		vec3 startOffset = dV_view*clamp(dither,0,1);
		progress_view = startOffset + cameraPosition + dV_view*(cloud_height-cameraPosition.y)/(dV_view.y);

		float mult = length(dV_view);


		color = vec3(0.0);
		float SdotV = dot(sunVec,normalize(fragpositi));
		//fake multiple scattering approx 1 (from horizon zero down clouds)
		float mieDay = max(phaseg(SdotV,0.4),phaseg(SdotV,0.2));
		float mieNight = max(phaseg(-SdotV,0.4),phaseg(-SdotV,0.2));

		vec3 sunContribution = mieDay*sunColor*3.14;
		vec3 moonContribution = mieNight*moonColor*3.14;
		float ambientMult = exp(-(1+0.24+0.8*clamp(rainStrength,0.75,1))*cdensity*50.0);
		vec3 skyCol0 = avgAmbient * ambientMult;
										  


		for (int i=0;i<maxIT_clouds;i++) {
		vec3 curvedPos = progress_view;
		vec2 xz = progress_view.xz-cameraPosition.xz;
		curvedPos.y -= sqrt(pow(6731e3,2.0)-dot(xz,xz))-6731e3;
		vec3 samplePos = curvedPos*vec3(1.0,1./32.,1.0)/4+frameTimeCounter*vec3(0.5,0.,0.5);
			 samplePos += vec3(10000,0,10000);
			float coverageSP = cloudCov(curvedPos,samplePos);
			if (coverageSP>0.05){
				float cloud = cloudVol(curvedPos,samplePos,coverageSP);
				if (cloud > 0.05){
					float mu = cloud*cdensity;
				


					//fake multiple scattering approx 2  (from horizon zero down clouds)
					float h = 0.35-0.35*clamp(progress_view.y/4000.-1500./4000.,0.0,1.0);
					float powder = 1.0-exp(-mu*mult);
					float Shadow =  mix(1.0, powder,  h);
					float ambientPowder = mix(1.0,powder,h * ambientMult);
					vec3 S = vec3(sunContribution*Shadow+Shadow*moonContribution+skyCol0*ambientPowder);


					vec3 Sint=(S - S * exp(-mult*mu)) / (mu);
					color += mu*Sint*total_extinction;
					total_extinction *= exp(-mu*mult);
					if (total_extinction < 1e-5) break;
				}
							 
			}

			progress_view += dV_view;

		}


		
		float cosY = normalize(dV_view).y;


			color.rgb = mix(color.rgb*vec3(0.5,0.5,1.0),color.rgb,1-rainStrength);	
		return mix(vec4(color,clamp(total_extinction*(1.0+1/250.)-1/250.,0.0,1.0)),vec4(0.0,0.0,0.0,1.0),1-smoothstep(0.02,0.7,cosY));

}
float luma(vec3 color){
	return dot(color,vec3(0.299, 0.587, 0.114));
}

float ditherGradNoise() {
  return fract(52.9829189 * fract(0.06711056 * gl_FragCoord.x + 0.00583715 * gl_FragCoord.y));
}

vec4 backProject(vec4 vec) {
    vec4 tmp = gbufferModelViewInverse * vec;
    return tmp / tmp.w;
}

// simplified version of joeedh's https://www.shadertoy.com/view/Md3GWf
// see also https://www.shadertoy.com/view/MdtGD7

// --- checkerboard noise : to decorelate the pattern between size x size tiles 

// simple x-y decorrelated noise seems enough
#define stepnoise0(p, size) rnd( floor(p/size)*size ) 
#define rnd(U) fract(sin( 1e3*(U)*mat2(1,-7.131, 12.9898, 1.233) )* 43758.5453)

//   joeedh's original noise (cleaned-up)
vec2 stepnoise(vec2 p, float size) { 
    p = floor((p+10.)/size)*size;          // is p+10. useful ?   
    p = fract(p*.1) + 1. + p*vec2(2,3)/1e4;    
    p = fract( 1e5 / (.1*p.x*(p.y+vec2(0,1)) + 1.) );
    p = fract( 1e5 / (p*vec2(.1234,2.35) + 1.) );      
    return p;    
}

// --- stippling mask  : regular stippling + per-tile random offset + tone-mapping

#define SEED1 1.705
#define DMUL  8.12235325       // are exact DMUL and -.5 important ?

float mask(vec2 p) { 

    p += ( stepnoise0(p, 5.5) - .5 ) *DMUL;   // bias [-2,2] per tile otherwise too regular
    float f = fract( p.x*SEED1 + p.y/(SEED1+.15555) ); //  weights: 1.705 , 0.5375

    //return f;  // If you want to skeep the tone mapping
    f *= 1.03; //  to avoid zero-stipple in plain white ?

    // --- indeed, is a tone mapping ( equivalent to do the reciprocal on the image, see tests )
    // returned value in [0,37.2] , but < 0.57 with P=50% 

    return  (pow(f, 150.) + 1.3*f ) / 2.3; // <.98 : ~ f/2, P=50%  >.98 : ~f^150, P=50%    
}    
vec2 sphereToCarte(vec3 dir) {
    float lonlat = atan(-dir.x, -dir.z);
    return vec2(lonlat * (0.5/PI) +0.5,0.5*dir.y+0.5);
}
void main() {

	float mod2 = gl_FragCoord.x + gl_FragCoord.y;
	float res = mod(mod2, 2.0f);


  {
    //vec3 rnd = ScreenSpaceDither( gl_FragCoord.xy );
    float noise = mask(gl_FragCoord.xy+(Time*100));
	vec2 halfResTC = vec2(floor(gl_FragCoord.xy)/CLOUDS_QUALITY+0.5);

    float aspectRatio = ScreenSize.x/ScreenSize.y;
        vec4 screenPos = gl_FragCoord;
        screenPos.xy = (screenPos.xy / ScreenSize - vec2(0.5)) * 2.0;
        screenPos.zw = vec2(1.0);


	 
    float depth = texture(DiffuseDepthSampler, texCoord).r;





    vec3 sc = texelFetch(temporals3Sampler,ivec2(8,37),0).rgb;
  	vec3 suncol = sc;

    vec2 scaledCoord = 2.0 * (halfResTC*oneTexel - vec2(0.5));
	bool doClouds = false;
	for (int i = 0; i < floor(1.0/CLOUDS_QUALITY)+1.0; i++){
		for (int j = 0; j < floor(1.0/CLOUDS_QUALITY)+1.0; j++){
			if (texelFetch(DiffuseDepthSampler,ivec2(halfResTC) + ivec2(i, j), 0).x >= 1.0)
				doClouds = true;
		}
	}
	if (doClouds){
    vec3 fragpos = backProject(vec4(scaledCoord, depth, 1.0)).xyz;
	vec4 cloud = renderClouds(fragpos,avgSky,noise,suncol,suncol,avgSky).rgba;

	fragColor = cloud;
	}
	else
		fragColor = vec4(0.0,0.0,0.0,1.0);

}

}
