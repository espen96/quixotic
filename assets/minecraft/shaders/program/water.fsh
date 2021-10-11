#version 150
out vec4 fragColor;

uniform sampler2D DiffuseSampler;
uniform sampler2D DiffuseDepthSampler;
uniform sampler2D TranslucentSampler;
uniform sampler2D TranslucentDepthSampler;
uniform sampler2D TerrainCloudsSampler;
uniform sampler2D temporals3Sampler;
uniform vec2 OutSize;
uniform vec2 ScreenSize;
in vec2 texCoord;
in vec2 oneTexel;
in float aspectRatio;
in float cosFOVrad;
in float tanFOVrad;

flat in vec3 ambientUp;
flat in vec3 ambientLeft;
flat in vec3 ambientRight;
flat in vec3 ambientB;
flat in vec3 ambientF;
flat in vec3 ambientDown;
flat in vec3 avgSky;
flat in float isEyeInLava;
flat in float isEyeInWater;
#define near 0.00004882812 
in float far;
in mat4 wgbufferModelViewInverse;

#define NORMDEPTHTOLERANCE 1.0
#define SSR_TAPS 3
#define SSR_SAMPLES 10
#define SSR_MAXREFINESAMPLES 10
#define SSR_STEPREFINE 0.2
#define SSR_STEPINCREASE 1.2
#define SSR_IGNORETHRESH 0.0
#define NORMAL_SCATTER 0.006








in vec2 texCoord2;

in vec3 sunDir;
in vec4 fogcol;
in mat4 gbufferModelViewInverse;
in mat4 gbufferModelView;
in mat4 gbufferProjection;
in mat4 gbufferProjectionInverse;


float GGX (vec3 n, vec3 v, vec3 l, float r, float F0) {
  r*=r;r*=r;

  vec3 h = l + v;
  float hn = inversesqrt(dot(h, h));

  float dotLH = clamp(dot(h,l)*hn,0.,1.);
  float dotNH = clamp(dot(h,n)*hn,0.,1.);
  float dotNL = clamp(dot(n,l),0.,1.);
  float dotNHsq = dotNH*dotNH;

  float denom = dotNHsq * r - dotNHsq + 1.;
  float D = r / (3.141592653589793 * denom * denom);
  float F = F0 + (1. - F0) * exp2((-5.55473*dotLH-6.98316)*dotLH);
  float k2 = .25 * r;

  return dotNL * D * F / (dotLH*dotLH*(1.0-k2)+k2);
}

  
float LinearizeDepth(float depth) 
{
    return (2.0 * near * far) / (far + near - depth * (far - near));    
}



float luminance(vec3 rgb) {
    float redness = clamp(dot(rgb, vec3(1.0, -0.25, -0.75)), 0.0, 1.0);
    return ((1.0 - redness) * dot(rgb, vec3(0.2126, 0.7152, 0.0722)) + redness * 1.4) * 4.0;
}

float luma4(vec3 color) {
	return dot(color,vec3(0.21, 0.72, 0.07));
}





vec2 unpackUnorm2x4(float pack) {
	vec2 xy; xy.x = modf(pack * 255.0 / 16.0, xy.y);
	return xy * vec2(16.0 / 15.0, 1.0 / 15.0);
}

////////////////////////////

// moj_import doesn't work in post-process shaders ;_; Felix pls fix
#define NUMCONTROLS 26
#define THRESH 0.5
#define FPRECISION 4000000.0
#define PROJNEAR 0.05
#define FUDGE 32.0



/////////
float invLinZ (float lindepth){
	return -((2.0*near/lindepth)-far-near)/(far-near);
}
float ld(float dist) {
    return (2.0 * near) / (far + near - dist * (far - near));
}
vec3 nvec3(vec4 pos){
    return pos.xyz/pos.w;
}

vec4 nvec4(vec3 pos){
    return vec4(pos.xyz, 1.0);
}
float linZ(float depth) {
    return (2.0 * near) / (far + near - depth * (far - near));
	// l = (2*n)/(f+n-d(f-n))
	// f+n-d(f-n) = 2n/l
	// -d(f-n) = ((2n/l)-f-n)
	// d = -((2n/l)-f-n)/(f-n)

}
#define diagonal3(m) vec3((m)[0].x, (m)[1].y, m[2].z)
#define  projMAD(m, v) (diagonal3(m) * (v) + (m)[3].xyz)

vec3 toClipSpace3(vec3 viewSpacePosition) {
    return projMAD(gbufferProjection, viewSpacePosition) / -viewSpacePosition.z * 0.5 + 0.5;
}
#define SSPTBIAS 0.9 //[0.0 0.001 0.01 0.015 0.025 0.05 0.1 0.125 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.9 1.0]


#define RAY_COUNT 2 // [1 2 3 4 5 6 7 8 9 10 12 14 16 18 21 24 28 32 37 43 49 57 65 75 86 100]
#define STEPS Unlimited	// [Unlimited 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99]
#define STEP_LENGTH 11.	// [ 4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15. 16. 17. 18. 19. 20. 21. 22. 23. 24. 25. 26. 27. 28. 29. 30. 60. 90.]
vec3 rayTrace(vec3 dir,vec3 position,float noise, float fresnel){
		//	float sp= linZ(texelFetch(TranslucentDepthSampler,ivec2(spos.xy/oneTexel),0).x);

	float ssptbias = SSPTBIAS;
	float stepSize = STEP_LENGTH;
#if STEPS != Unlimited
	int maxSteps = STEPS;
#endif





	vec3 clipPosition = toClipSpace3(position);
	
	float rayLength = ((position.z + dir.z * sqrt(3.0)*far) > -sqrt(3.0)*near) ? (-sqrt(3.0)*near -position.z) / dir.z : sqrt(3.0)*far;

	vec3 end = toClipSpace3(position+dir*rayLength);
	vec3 direction = end-clipPosition;  //convert to clip space

	float len = max(abs(direction.x)/oneTexel.x,abs(direction.y)/oneTexel.y)/stepSize;

	//get at which length the ray intersects with the edge of the screen
	vec3 maxLengths = (step(0.,direction)-clipPosition) / direction;
	float mult = min(min(maxLengths.x,maxLengths.y),maxLengths.z);
	vec3 stepv =direction * mult / 15;
//    vec3 stepv = direction/len;

#if STEPS == Unlimited
	int iterations = int(min(len, mult*len)-2);
#else	
	int iterations = min(int(min(len, mult*len)-2), maxSteps);
#endif
	
	//Do one iteration for closest texel (good contact shadows)
	vec3 spos = clipPosition*vec3(1.0) + stepv/stepSize*6.0;

	spos.xy += 0*oneTexel*0.5;

	float sp= linZ(texelFetch(TranslucentDepthSampler,ivec2(spos.xy/oneTexel),0).x);
	float currZ = linZ(spos.z);

	

	stepv *= vec3(1.0);

		
	spos += stepv*noise;

 //   for(int i = 0; i < iterations; i++){
    for(int i = 0; i < 15; i++){
        if (clamp(clipPosition.xy,0,1) != clipPosition.xy) break;
		// decode depth buffer
		float sp= linZ(texelFetch(TranslucentDepthSampler,ivec2(spos.xy/oneTexel),0).x);
			
		float currZ = linZ(spos.z);
	//	if( sp < currZ && abs(sp-ld(spos.z))/ld(spos.z) < 0.1) {
		if( sp < currZ -0.00001) {
			if (spos.x < 0.0 || spos.y < 0.0 || spos.z < 0.0 || spos.x > 1.0 || spos.y > 1.0 || spos.z > 1.0) return vec3(1.1);
			float dist = abs(sp-currZ)/currZ;

			if (dist <= ssptbias) return vec3(spos.xy, invLinZ(sp))/vec3(1.0);

		}
		

			spos += stepv;	

	}




	return vec3(1.1);

	
}

vec3 skyLut(vec3 sVector, vec3 sunVec,float cosT,sampler2D lut) {
	const vec3 moonlight = vec3(0.8, 1.1, 1.4) * 0.06;

	float mCosT = clamp(cosT,0.0,1.);
	float cosY = dot(sunVec,sVector);
	float x = ((cosY*cosY)*(cosY*0.5*256.)+0.5*256.+18.+0.5)*oneTexel.x;
	float y = (mCosT*256.+1.0+0.5)*oneTexel.y;

	return texture(lut,vec2(x,y)).rgb;


}
float ditherGradNoise() {
  return fract(52.9829189 * fract(0.06711056 * gl_FragCoord.x + 0.00583715 * gl_FragCoord.y));
}
float R2_dither(){
	vec2 alpha = vec2(0.75487765, 0.56984026);
	return fract(alpha.x * gl_FragCoord.x + alpha.y * gl_FragCoord.y);
}

vec3 toScreenSpace(vec3 p) {
	vec4 iProjDiag = vec4(gbufferProjectionInverse[0].x, gbufferProjectionInverse[1].y, gbufferProjectionInverse[2].zw);
    vec3 p3 = p * 2. - 1.;
    vec4 fragposition = iProjDiag * p3.xyzz + vec4(0,0,gbufferProjectionInverse[3].ba);
    return fragposition.xyz / fragposition.w;
}
vec4 backProject(vec4 vec) {
    vec4 tmp = wgbufferModelViewInverse * vec;
    return tmp / tmp.w;
}

vec3 worldToView(vec3 worldPos) {

    vec4 pos = vec4(worldPos, 0.0);
    pos = gbufferModelView * pos +gbufferModelView[3];

    return pos.xyz;
}

void main() {


    vec3 reflection = vec3(1.0);
    

    vec4 color = texture(TranslucentSampler, texCoord);


    vec4 color2 = color;

    if (color.a > 0.01 ) {
    float mod2 = gl_FragCoord.x + gl_FragCoord.y;
    float res = mod(mod2, 2.0f);
    vec2 lmtrans = unpackUnorm2x4((texture(DiffuseSampler, texCoord).a));
    vec2 lmtrans3 = unpackUnorm2x4((texture(DiffuseSampler, texCoord+oneTexel.y).a));

    float lmx = 0;
    float lmy = 0;


          lmy = mix(lmtrans.y,lmtrans3.y,res);
          lmx = clamp(mix(lmtrans3.y,lmtrans.y,res)*2.0,0,1);
    float lm = clamp(lmx-lmy,0,1);


    vec2 poissonDisk[16];
    poissonDisk[0] = vec2(-0.613392, 0.617481);
    poissonDisk[1] = vec2(0.170019, -0.040254);
    poissonDisk[2] = vec2(-0.299417, 0.791925);
    poissonDisk[3] = vec2(0.645680, 0.493210);
    poissonDisk[4] = vec2(-0.651784, 0.717887);
    poissonDisk[5] = vec2(0.421003, 0.027070);
    poissonDisk[6] = vec2(-0.817194, -0.271096);
    poissonDisk[7] = vec2(-0.705374, -0.668203);
    poissonDisk[8] = vec2(0.977050, -0.108615);
    poissonDisk[9] = vec2(0.063326, 0.142369);
    poissonDisk[10] = vec2(0.203528, 0.214331);
    poissonDisk[11] = vec2(-0.667531, 0.326090);
    poissonDisk[12] = vec2(-0.098422, -0.295755);
    poissonDisk[13] = vec2(-0.885922, 0.215369);
    poissonDisk[14] = vec2(0.566637, 0.605213);
    poissonDisk[15] = vec2(0.039766, -0.396100);


    vec3 sky = mix(color.rgb*2.0,avgSky,0.5);

    float deptha = texture(TranslucentDepthSampler, texCoord).r;
    float depthb = texture(TranslucentDepthSampler, texCoord-vec2(0,oneTexel.y)).r;
    float depthc = texture(TranslucentDepthSampler, texCoord+vec2(0,oneTexel.y)).r;
    float depthd = texture(TranslucentDepthSampler, texCoord+vec2(oneTexel.x,0)).r;
    float depthe = texture(TranslucentDepthSampler, texCoord-vec2(oneTexel.x,0)).r;

        float depth2 = depthc;
        float depth3 = depthd;
        float depth4 = depthb;
        float depth5 = depthe;
        #define normalstrength  0.1;    
        float normaldistance = 2.5;    
        float normalpow = 4.0;    
        vec2 scaledCoord = 2.0 * (texCoord - vec2(0.5));
        vec3 fragpos = backProject(vec4(scaledCoord, deptha, 1.0)).xyz;

        vec3 p2 = backProject(vec4(scaledCoord + 2.0 * vec2(0.0, oneTexel.y), depth2, 1.0)).xyz;
        p2 = p2 - fragpos;

        vec3 p3 = backProject(vec4(scaledCoord + 2.0 * vec2(oneTexel.x, 0.0), depth3, 1.0)).xyz;
        p3 = p3 - fragpos;

        vec3 p4 = backProject(vec4(scaledCoord - 2.0 * vec2(0.0, oneTexel.y), depth4, 1.0)).xyz;
        p4 = p4 - fragpos;

        vec3 p5 = backProject(vec4(scaledCoord - 2.0 * vec2(oneTexel.x, 0.0), depth5, 1.0)).xyz;
        p5 = p5 - fragpos;

        vec3 normal = normalize(cross( p2,  p3)) 
                    + normalize(cross(-p4,  p3)) 
                    + normalize(cross( p2, -p5)) 
                    + normalize(cross(-p4, -p5));
        normal = normal == vec3(0.0) ? vec3(0.0, 1.0, 0.0) : normalize(-normal);
        vec3 normal2 = normal;
        normal = worldToView (normal);

////////////////////
    float z = texture(TranslucentDepthSampler,texCoord).x;
    vec3 fragpos3 = toScreenSpace(vec3(texCoord-vec2(0)*oneTexel*0.5,z));


		float f0 = 0.02;

		float roughness = 0.02;

		float emissive = 0.0;
		float F0 = f0;

		vec3 reflectedVector = reflect(normalize(fragpos3), normal);
		float normalDotEye = dot(normal, normalize(fragpos3));
		float fresnel2 = pow(clamp(1.0 + normalDotEye,0.0,1.0), 5.0);
		fresnel2 = mix(F0,1.0,fresnel2);
	
			fresnel2 = fresnel2*0.87+0.04;	//faking additionnal roughness to the water
			roughness = 0.1;
		

    vec4 screenPos = gl_FragCoord;
         screenPos.xy = (screenPos.xy / ScreenSize - vec2(0.5)) * 2.0;
         screenPos.zw = vec2(1.0);
    vec3 view = normalize((wgbufferModelViewInverse * screenPos).xyz);
    vec3 view2 = view;
         view2.y = -view2.y;

		vec3 sky_c = avgSky.rgb;
            vec3 suncol = texelFetch(temporals3Sampler,ivec2(8,37),0).rgb*0.7;

    sky_c = ((skyLut(view2,sunDir.xyz,view2.y,temporals3Sampler))*suncol)  ;




		vec4 reflection = vec4(sky_c.rgb,0.);

		vec3 rtPos = rayTrace(reflectedVector,fragpos3.xyz,R2_dither(), fresnel2);
		if (rtPos.z <1.){
	    vec4 fragpositionPrev = gbufferProjectionInverse * vec4(rtPos*2.-1.,1.);
		fragpositionPrev /= fragpositionPrev.w;
	    reflection.a = 1.0;
		reflection.rgb = texture(TerrainCloudsSampler,rtPos.xy).rgb;
		}

        float sunSpec = ((GGX(normal2,-normalize(view),  sunDir, roughness, F0)))*0;		

		reflection.rgb = mix(sky_c.rgb, reflection.rgb, reflection.a);

        vec3 reflected= reflection.rgb*fresnel2+1.0*sunSpec;

        float alpha0 = color2.a;
	    color.a = -color2.a*fresnel2+color2.a+fresnel2;
		color.rgb =clamp((color2.rgb*8)/color.a*alpha0*(1.0-fresnel2)*0.1+(reflected*10)/color.a*0.1,0.0,1.0);
//        color.rgb = vec3(sky_c);

    }        
   

    fragColor= vec4(color.rgba);
}
