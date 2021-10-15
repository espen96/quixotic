#version 150
out vec4 fragColor;

uniform sampler2D DiffuseSampler;
uniform sampler2D DiffuseDepthSampler;
uniform sampler2D TranslucentSampler;
uniform sampler2D TranslucentDepthSampler;
uniform sampler2D TerrainCloudsSampler;
uniform sampler2D temporals3Sampler;
uniform vec2 ScreenSize;
uniform float Time;
in vec2 texCoord;
in vec2 oneTexel;


in float near;
in float far;

in vec3 sunDir;
in mat4 gbufferModelView;
in mat4 gbufferProjection;
in mat4 gbufferProjectionInverse;
in mat4 wgbufferModelViewInverse;


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
}
#define diagonal3(m) vec3((m)[0].x, (m)[1].y, m[2].z)
#define  projMAD(m, v) (diagonal3(m) * (v) + (m)[3].xyz)


#define  projMAD2(m, v) (diagonal3(m) * (v) + vec3(0,0,m[3].b))

vec3 toClipSpace3(vec3 viewSpacePosition) {
    return projMAD2(gbufferProjection, viewSpacePosition) / -viewSpacePosition.z * 0.5 + 0.5;
}

#define SSPTBIAS 0.5

#define SSR_STEPS 20 //[10 15 20 25 30 35 40 50 100 200 400]
float hash12(vec2 p)
{
	vec3 p3  = fract(vec3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

vec3 skyLut(vec3 sVector, vec3 sunVec,float cosT,sampler2D lut) {
	const vec3 moonlight = vec3(0.8, 1.1, 1.4) * 0.06;

	float mCosT = clamp(cosT,0.0,1.);
	float cosY = dot(sunVec,sVector);
	float x = ((cosY*cosY)*(cosY*0.5*256.)+0.5*256.+18.+0.5)*oneTexel.x;
	float y = (mCosT*256.+1.0+0.5)*oneTexel.y;

	return texture(lut,vec2(x,y)).rgb;


}

float R2_dither(){
	vec2 alpha = vec2(0.75487765, 0.56984026);
	return fract(alpha.x * gl_FragCoord.x + alpha.y * gl_FragCoord.y + 1.0/1.6180339887 * Time);
}

vec3 toScreenSpace(vec3 p) {
	vec4 iProjDiag = vec4(gbufferProjectionInverse[0].x, gbufferProjectionInverse[1].y, gbufferProjectionInverse[2].zw);
    vec3 p3 = p * 2. - 1.;
    vec4 fragposition = iProjDiag * p3.xyzz + vec4(0,0,gbufferProjectionInverse[3].b,gbufferProjectionInverse[3].a);
    return fragposition.xyz / fragposition.w;
}

vec3 worldToView(vec3 worldPos) {

    vec4 pos = vec4(worldPos, 0.0);
    pos = gbufferModelView * pos +gbufferModelView[3];

    return pos.xyz;
}


vec3 getDepthPoint(vec2 coord, float depth) {
    vec4 pos;
    pos.xy = coord;
    pos.z = depth;
    pos.w = 1.0;
    pos.xyz = pos.xyz * 2.0 - 1.0; //convert from the 0-1 range to the -1 to +1 range
    pos = gbufferProjectionInverse * pos;
    pos.xyz /= pos.w;
    
    return pos.xyz;
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

vec3 constructNormal(float depthA, vec2 texcoords, sampler2D depthtex) {
     vec2 offsetB = vec2(0.0,oneTexel.y);
     vec2 offsetC = vec2(oneTexel.x,0.0);
  
    float depthB = texture(depthtex, texcoords + offsetB).r;
    float depthC = texture(depthtex, texcoords + offsetC).r;
  
    vec3 A = getDepthPoint(texcoords, depthA);
	vec3 B = getDepthPoint(texcoords + offsetB, depthB);
	vec3 C = getDepthPoint(texcoords + offsetC, depthC);

	vec3 AB = normalize(B - A);
	vec3 AC = normalize(C - A);

	vec3 normal =  -cross(AB, AC);
	// normal.z = -normal.z;

	return normalize(normal);
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

//Dithering from Jodie
float Bayer2(vec2 a) {
    a = floor(a+(Time));
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

float cdist(vec2 coord) {
	return max(abs(coord.s-0.5) * 1.95, abs(coord.t-0.5) * 2.0);
}

vec4 Raytrace(sampler2D depthtex, vec3 viewPos, vec3 normal, float dither, out float border, 
			  float maxf, float stp, float ref, float inc) {

	vec3 pos = vec3(0.0);
	float dist = 0.0;
    int sr = 0;



	vec3 start = viewPos;

    vec3 vector = stp * reflect(normalize(viewPos), normalize(normal));
    viewPos += vector;
	vec3 tvector = vector;



    for(int i = 0; i < 12; i++) {
        pos = nvec3(gbufferProjection * nvec4(viewPos)) * 0.5 + 0.5;
		if (pos.x < -0.05 || pos.x > 1.05 || pos.y < -0.05 || pos.y > 1.05) break;

		vec3 rfragpos = vec3(pos.xy, texture(DiffuseDepthSampler,pos.xy).r);
        rfragpos = nvec3(gbufferProjectionInverse * nvec4(rfragpos * 2.0 - 1.0));
		dist = length(start - rfragpos);

        float err = length(viewPos - rfragpos);
		float lVector = length(vector);
		if (lVector > 1.0) lVector = pow(lVector, 1.14);
		if (err < lVector) {
                sr++;
			if (sr >= maxf) break;
			tvector -= vector;
			vector *= ref;
		}
        vector *= inc;
        tvector += vector;
		viewPos = start + tvector * (dither * 0.05 + 0.975);
    }

	border = cdist(pos.st);


	return vec4(pos, dist);
}
vec4 SSR(vec3 fragpos, float fragdepth, vec3 surfacenorm, vec4 skycol, float noise) {

    vec3 pos    = vec3(0.0);

float maxf = 1.0; // 4.0 max refinement steps
float stp  = 1.0; // 1.0 initial length of the reflected vector
float ref  = 0.1; // 0.1 refinement multiplier
float inc  = 2.0; // 2.0 iteration multiplier



    vec4 color = vec4(0.0);
	float border = 0.0;
     pos = Raytrace(DiffuseDepthSampler, fragpos, surfacenorm, noise , border, maxf, stp, ref, inc).xyz;

	border = clamp(13.333 * (1.0 - border), 0.0, 1.0);
	
	if (pos.z < 1.0 - 1e-5) {
		color.a = texture(TerrainCloudsSampler, pos.st).a;
		if (color.a > 0.001) {color.rgb = texture(TerrainCloudsSampler, pos.st).rgb;
		color.rgb += texture(TranslucentSampler, pos.st).rgb*0.45;}
        color.rgb *= border;
		
		color.a *= border;
	}

    return color;
}
void main() {


    vec3 reflection = vec3(1.0);
    

    vec4 color = texture(TranslucentSampler, texCoord);


    vec4 color2 = color;

    if (color.a > 0.01 ) {




    float depth = texture(TranslucentDepthSampler, texCoord).r;

    vec3 normal = constructNormal(depth, texCoord,  TranslucentDepthSampler);


////////////////////
    vec3 fragpos3 = toScreenSpace(vec3(texCoord,depth));
    vec3 screenPos2 = vec3(texCoord, depth);
    vec3 clipPos = screenPos2 * 2.0 - 1.0;
    vec4 tmp = gbufferProjectionInverse * vec4(clipPos, 1.0);
    vec3 viewPos = tmp.xyz / tmp.w;	
    


		float roughness = 0.1;

		float F0 = 0.02;

		float normalDotEye = dot(normal, normalize(fragpos3));
		float fresnel = pow(clamp(1.0 + normalDotEye,0.0,1.0), 5.0)*0.87+0.04;
		fresnel = mix(F0,1.0,fresnel);
        float noise = mask(gl_FragCoord.xy+(Time*100));
		

        vec4 screenPos = gl_FragCoord;
             screenPos.xy = (screenPos.xy / ScreenSize - vec2(0.5)) * 2.0;
             screenPos.zw = vec2(1.0);
        vec3 view = normalize((wgbufferModelViewInverse * screenPos).xyz);
        vec3 view2 = view;
             view2.y = -view2.y;

        vec3 suncol = texelFetch(temporals3Sampler,ivec2(8,37),0).rgb*0.5;

        vec3 sky_c = (mix(skyLut(view2,sunDir.xyz,view2.y,temporals3Sampler),suncol,0.5))  ;




		vec4 reflection = vec4(sky_c.rgb,0.);

        normal += noise*0.02;
        reflection = vec4(SSR(viewPos.xyz, depth,normal, vec4(sky_c,1), noise ));	
        reflection.rgb = mix(sky_c.rgb, reflection.rgb, reflection.a);
        vec3 reflected= reflection.rgb*fresnel;

        float alpha0 = color2.a;
	    color.a = -color2.a*fresnel+color2.a+fresnel;
		color.rgb =clamp((color2.rgb*6.5)/color.a*alpha0*(1.0-fresnel)*0.1+(reflected*10)/color.a*0.1,0.0,1.0);
    //    color.rgb = normal;

    }        
   
   
    fragColor= vec4(color.rgba);
}
