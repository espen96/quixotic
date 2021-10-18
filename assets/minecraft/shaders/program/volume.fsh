#version 150

uniform sampler2D MainSampler;
uniform sampler2D DiffuseDepthSampler;
uniform sampler2D temporals3Sampler;
uniform sampler2D TranslucentSampler;
uniform sampler2D TranslucentDepthSampler;



uniform vec2 ScreenSize;
uniform float Time;




in vec2 texCoord;
in vec2 oneTexel;
flat in vec4 fogcol;
flat in vec4 skycol;
in vec4 rain;
in mat4 gbufferModelViewInverse;
flat in float near;
flat in float far;
flat in float end;
flat in float overworld;
flat in vec3 currChunkOffset;

flat in float sunElevation;
flat in vec3 sunVec;
flat in vec3 sunPosition;
flat in float fogAmount;
flat in vec2 eyeBrightnessSmooth;
in vec3 avgSky;

#define VL_SAMPLES 4 
#define Ambient_Mult 1.0 
#define SEA_LEVEL 70 //The volumetric light uses an altitude-based fog density, this is where fog density is the highest, adjust this value according to your world.
#define ATMOSPHERIC_DENSITY 1.0
#define fog_mieg1 0.40
#define fog_mieg2 0.10
#define fog_coefficientRayleighR 5.8
#define fog_coefficientRayleighG 1.35
#define fog_coefficientRayleighB 3.31

#define fog_coefficientMieR 2.0
#define fog_coefficientMieG 5.0
#define fog_coefficientMieB 10.0



#define Dirt_Amount 0.015  //How much dirt there is in water

#define Dirt_Scatter_R 0.6  //How much dirt diffuses red
#define Dirt_Scatter_G 0.6  //How much dirt diffuses green
#define Dirt_Scatter_B 0.6  //How much dirt diffuses blue

#define Dirt_Absorb_R 0.65  //How much dirt absorbs red
#define Dirt_Absorb_G 0.85  //How much dirt absorbs green
#define Dirt_Absorb_B 1.05  //How much dirt absorbs blue

#define Water_Absorb_R 0.25422  //How much water absorbs red
#define Water_Absorb_G 0.03751  //How much water absorbs green
#define Water_Absorb_B 0.01150  //How much water absorbs blue

#define Dirt_Mie_Phase 0.4  //Values close to 1 will create a strong peak around the sun and weak elsewhere, values close to 0 means uniform fog.




out vec4 fragColor;
const float pi = 3.141592653589793238462643383279502884197169;




vec3 suncol = texelFetch(temporals3Sampler,ivec2(8,37),0).rgb*10;



vec4 lightCol = vec4(suncol,float(sunElevation > 1e-5)*2-1.);



float LinearizeDepth(float depth) 
{
    return (2.0 * near * far) / (far + near - depth * (far - near));    
}

float luma(vec3 color){
	return dot(color,vec3(0.299, 0.587, 0.114));
}

vec4 backProject(vec4 vec) {
    vec4 tmp = gbufferModelViewInverse * vec;
    return tmp / tmp.w;
}




vec3 normVec (vec3 vec){
	return vec*inversesqrt(dot(vec,vec));
}

float ditherGradNoise() {
  return fract(52.9829189 * fract(0.06711056 * gl_FragCoord.x + 0.00583715 * gl_FragCoord.y));
}



float GetLinearDepth(float depth) {
   return (2.0 * near) / (far + near - depth * (far - near));
}



float packUnorm2x4(vec2 xy) {
	return dot(floor(15.0 * xy + 0.5), vec2(1.0 / 255.0, 16.0 / 255.0));
}
float packUnorm2x4(float x, float y) { return packUnorm2x4(vec2(x, y)); }
vec2 unpackUnorm2x4(float pack) {
	vec2 xy; xy.x = modf(pack * 255.0 / 16.0, xy.y);
	return xy * vec2(16.0 / 15.0, 1.0 / 15.0);
}



///////////////////////////////////

float facos(float inX) {

	const float C0 = 1.56467;
	const float C1 = -0.155972;

    float x = abs(inX);
    float res = C1 * x + C0;
    res *= sqrt(1.0f - x);

    return (inX >= 0) ? res : pi - res;
}

float R2_dither(){
	vec2 alpha = vec2(0.75487765, 0.56984026);
	return fract(alpha.x * gl_FragCoord.x + alpha.y * gl_FragCoord.y + 0.43015971 * Time);
}
float phaseRayleigh(float cosTheta) {
	 vec2 mul_add = vec2(0.1, 0.28) /facos(-1.0);
	return cosTheta * mul_add.x + mul_add.y; // optimized version from [Elek09], divided by 4 pi for energy conservation
}
float phaseg(float x, float g){
    float gg = g * g;
    return (gg * -0.25 + 0.25) * pow(-2.0 * (g * x) + (gg + 1.0), -1.5) /3.1415;
}
float cloudVol(in vec3 pos){
	float unifCov = exp2(-max(pos.y-SEA_LEVEL,0.0)/50.);
	float cloud = unifCov*60.*fogAmount;
  return cloud;
}


mat2x3 getVolumetricRays(float dither,vec3 fragpos, vec3 ambientUp, float fogv) {




//vec2 eyeBrightnessSmooth = vec2(0,( 0 + (sunElevation - -1) * (240 - 0) / (1 - -1)));



ambientUp = ambientUp*10;

//	vec3 wpos = mat3(gbufferModelViewInverse) * fragpos + gbufferModelViewInverse[3].xyz;
	vec3 wpos = fragpos;
	vec3 dVWorld = (wpos-gbufferModelViewInverse[3].xyz);

	float maxLength = min(length(dVWorld),far)/length(dVWorld);
	dVWorld *= maxLength;

	vec3 progressW = gbufferModelViewInverse[3].xyz+currChunkOffset;
	vec3 vL = vec3(0.);

	float SdotV = dot(sunPosition,normalize(fragpos))*lightCol.a;
	float dL = length(dVWorld);
	//Mie phase + somewhat simulates multiple scattering (Horizon zero down cloud approx)
	float mie = max(phaseg(SdotV,fog_mieg1),1.0/13.0);
	float rayL = phaseRayleigh(SdotV);
//	wpos.y = clamp(wpos.y,0.0,1.0);

	vec3 ambientCoefs = dVWorld/dot(abs(dVWorld),vec3(1.));


	vec3 ambientLight = avgSky;

	vec3 skyCol0 = ambientLight*8.*2./150./3.*eyeBrightnessSmooth.y/vec3(240.)*Ambient_Mult/3.1415;
	vec3 sunColor = lightCol.rgb*8./5./3.;
	sunColor *= 1-((1-rain.x)*0.5);
	skyCol0 *= 1-((1-rain.x)*0.5);

		vec3 rC = vec3(fog_coefficientRayleighR*1e-6, fog_coefficientRayleighG*1e-5, fog_coefficientRayleighB*1e-5);
	//	vec3 mC = vec3(fog_coefficientMieR*1e-6, fog_coefficientMieG*1e-6, fog_coefficientMieB*1e-6);
    vec4 skycol = skycol *3.0; 
		vec3 mC = vec3(skycol.r*1e-6, skycol.g*1e-6,skycol.b*1e-6);




	float mu = 1.0;
	float muS = 1.0*mu;
	vec3 absorbance = vec3(1.0);
	float expFactor = 2.7;
	for (int i=0;i<VL_SAMPLES;i++) {
		float d = (pow(expFactor, float(i+dither)/float(VL_SAMPLES))/expFactor - 1.0/expFactor)/(1-1.0/expFactor);
		float dd = pow(expFactor, float(i+dither)/float(VL_SAMPLES)) * log(expFactor) / float(VL_SAMPLES)/(expFactor-1.0);
		progressW = gbufferModelViewInverse[3].xyz+0 + d*dVWorld;
    	float density = cloudVol(progressW)*1.5*ATMOSPHERIC_DENSITY*mu*600.;
		//Just air
		vec2 airCoef = exp2(-max(progressW.y-SEA_LEVEL,0.0)/vec2(8.0e3, 1.2e3)*vec2(6.,7.0))*6.0;

		//Pbr for air, yolo mix between mie and rayleigh for water droplets
		vec3 rL = rC*(airCoef.x+density*0.15);
		vec3 m = (airCoef.y+density*1.85)*mC;
		vec3 vL0 = sunColor*(rayL*rL+m*mie)*0.75 + skyCol0*(rL+m);
		vL += vL0 * dd * dL *  absorbance;
		absorbance *= exp(-(rL+m)*dL*dd);
	}
    float lumC = luma(vL);
	vec3 diff = vL-lumC;
//	vL = vL + diff*(-lumC*2.0 + 0.6);

	return mat2x3(vL,absorbance);
}

void waterVolumetrics(inout vec3 inColor, vec3 rayStart, vec3 rayEnd, float estEyeDepth, float estSunDepth, float rayLength, float dither, vec3 waterCoefs, vec3 scatterCoef, vec3 ambient, vec3 lightSource, float VdotL, float sunElevation){
		int spCount = 6;
		//limit ray length at 32 blocks for performance and reducing integration error
		//you can't see above this anyway
		float maxZ = min(rayLength,32.0)/(1e-8+rayLength);
		rayLength *= maxZ;
		float dY = normalize(rayEnd).y * rayLength;
		vec3 absorbance = vec3(1.0);
		vec3 vL = vec3(0.0);
		float phase = phaseg(VdotL, Dirt_Mie_Phase);
		float expFactor = 11.0;
		for (int i=0;i<spCount;i++) {
			float d = (pow(expFactor, float(i+dither)/float(spCount))/expFactor - 1.0/expFactor)/(1-1.0/expFactor);		// exponential step position (0-1)
			float dd = pow(expFactor, float(i+dither)/float(spCount)) * log(expFactor) / float(spCount)/(expFactor-1.0);	//step length (derivative)
			vec3 ambientMul = exp(-max(estEyeDepth - dY * d,0.0) * waterCoefs * 1.1);
			vec3 sunMul = exp(-max((estEyeDepth - dY * d) ,0.0)/abs(sunElevation) * waterCoefs);
			vec3 light = (0.75 * lightSource * phase * sunMul + ambientMul*ambient )*scatterCoef;
			vL += (light - light * exp(-waterCoefs * dd * rayLength)) / waterCoefs *absorbance;
			absorbance *= exp(-dd * rayLength * waterCoefs);
		}
		inColor += vL;
}




void main() {
    float depth = texture(DiffuseDepthSampler, texCoord).r;

  	vec2 texCoord = texCoord; 
  	vec2 texCoord2 = texCoord; 
  float lum = luma(fogcol.rgb);
  vec3 diff = fogcol.rgb-lum;
  vec3 test  = clamp(vec3(0.0) + diff*(-lum*1.0 + 2),0,1);
 int isEyeInWater = 0;
 int isEyeInLava = 0;
 if(fogcol.a > 0.078 && fogcol.a < 0.079 ) isEyeInWater = 1;
 if(fogcol.r ==0.6 && fogcol.b == 0.0 ) isEyeInLava = 1;



    depth = texture(TranslucentDepthSampler, texCoord).r;
	vec3 vl = vec3(0.);

    vec4 screenPos = gl_FragCoord;
         screenPos.xy = (screenPos.xy / ScreenSize - vec2(0.5)) * 2.0;
         screenPos.zw = vec2(1.0);
    vec3 view = normalize((gbufferModelViewInverse * screenPos).xyz);


    vec3 OutTexel = texture(MainSampler, texCoord).rgb;
    vec2 scaledCoord = 2.0 * (texCoord - vec2(0.5));

    vec3 fragpos = backProject(vec4(scaledCoord, depth, 1.0)).xyz;
    fragColor.rgb = OutTexel;	

    float mod2 = gl_FragCoord.x + gl_FragCoord.y;
    float res = mod(mod2, 2.0f);
    vec2 lmtrans = unpackUnorm2x4((texture(MainSampler, texCoord2).a));
    vec2 lmtrans3 = unpackUnorm2x4((texture(MainSampler, texCoord2+oneTexel.y).a));

    float lmx = 0;
    float lmy = 0;
          lmy = mix(lmtrans.y,lmtrans3.y,res);
          lmx = mix(lmtrans3.y,lmtrans.y,res);
    if(depth >=1.0) lmx = 1.0;

	if(overworld == 1.0){





    float al = length(OutTexel);

    vec3 direct;
    vec3 ambient;
    direct = suncol;		
    





    float df = length(fragpos) ;


   if (isEyeInWater == 1 && overworld == 1){


       
      float dirtAmount = Dirt_Amount;
      vec3 waterEpsilon = vec3(Water_Absorb_R, Water_Absorb_G, Water_Absorb_B)*fogcol.rgb;
      vec3 dirtEpsilon = vec3(Dirt_Absorb_R, Dirt_Absorb_G, Dirt_Absorb_B);
      vec3 totEpsilon = dirtEpsilon*dirtAmount + waterEpsilon;
      vec3 scatterCoef = dirtAmount * vec3(Dirt_Scatter_R, Dirt_Scatter_G, Dirt_Scatter_B) ;
      fragColor.rgb *= clamp(exp(-df*totEpsilon),0.2,1.0);
      float estEyeDepth = clamp((14.0-(lmx*240)/255.0*16.0)/14.0,0.,1.0);
      estEyeDepth *= estEyeDepth*estEyeDepth*2.0;
    

     waterVolumetrics(vl, vec3(0.0), fragpos, estEyeDepth, estEyeDepth, length(fragpos), ditherGradNoise(), totEpsilon, scatterCoef, avgSky, direct.rgb, dot(normalize(fragpos), normalize(sunPosition)),sunElevation);

	  fragColor.rgb += vl;
      if(depth >=1)fragColor.rgb = vec3(vl);
    }


   else if (isEyeInWater == 0 ){
      mat2x3 vl = getVolumetricRays(R2_dither(),fragpos,avgSky,sunElevation);
     fragColor.rgb *= vl[1];
     fragColor.rgb += vl[0];
     if(luma(texture(TranslucentSampler, texCoord).rgb) > 0.0)lmx =0.93;
     lmx += LinearizeDepth(depth)*0.005;
     fragColor.rgb = mix( OutTexel,fragColor.rgb,clamp(lmx,0,1) );

     fragColor.a = vl[1].r;
    }
	}
    
	else {fragColor.rgb =  mix(fragColor.rgb*2.0,fogcol.rgb*0.5,pow(depth,256));
     fragColor.a = pow(depth,256);

	}


  
   if (isEyeInLava == 1 ){	 

    fragColor.rgb *= exp(-length(fragpos)*vec3(0.2,0.7,4.0)*4.);
    fragColor.rgb += vec3(4.0,0.5,0.1)*0.5;
   }
   if (fogcol.r == 0 && fogcol.g == 0 && fogcol.b == 0){	 

    fragColor.rgb *= exp(-length(fragpos)*vec3(1.0)*0.25);
   }
//  fragColor = vec4(vec3(lmx),1);

}
