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
}
#define diagonal3(m) vec3((m)[0].x, (m)[1].y, m[2].z)
#define  projMAD(m, v) (diagonal3(m) * (v) + (m)[3].xyz)

vec3 toClipSpace3(vec3 viewSpacePosition) {
    return projMAD(gbufferProjection, viewSpacePosition) / -viewSpacePosition.z * 0.5 + 0.5;
}
#define SSPTBIAS 0.2

#define SSR_STEPS 20 //[10 15 20 25 30 35 40 50 100 200 400]

vec3 rayTrace(vec3 dir,vec3 position,float noise, float fresnel){


	float ssptbias = SSPTBIAS;

    float quality = mix(10,SSR_STEPS,fresnel);
    vec3 clipPosition = toClipSpace3(position);
	float rayLength = ((position.z + dir.z * far*sqrt(3.)) > -near) ? (-near -position.z) / dir.z : far*sqrt(3.);

    vec3 direction = normalize(toClipSpace3(position+dir*rayLength)-clipPosition);  //convert to clip space
    direction.xy = normalize(direction.xy);

    //get at which length the ray intersects with the edge of the screen
    vec3 maxLengths = (step(0.,direction)-clipPosition) / direction;
    float mult = min(min(maxLengths.x,maxLengths.y),maxLengths.z);

	vec3 stepv =direction * mult / quality;
	vec3 spos = clipPosition *1.0 + stepv*noise;
	spos += stepv*noise;


    for(int i = 0; i < quality; i++){
        if (clamp(clipPosition.xy,0,1) != clipPosition.xy) break;

		float sp= linZ(texelFetch(TranslucentDepthSampler,ivec2(spos.xy/oneTexel),0).x);
			
		float currZ = linZ(spos.z);
	    if( sp < currZ -0.003) {
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

float R2_dither(){
	vec2 alpha = vec2(0.75487765, 0.56984026);
	return fract(alpha.x * gl_FragCoord.x + alpha.y * gl_FragCoord.y + 1.0/1.6180339887 * Time);
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

vec3 reconstructPosition(in vec2 uv, in float z, in mat4 InvVP)
{
  float x = uv.x * 2.0f - 1.0f;
  float y = (1.0 - uv.y) * 2.0f - 1.0f;
  vec4 position_s = vec4(x, y, z, 1.0f);
  vec4 position_v = InvVP*position_s;
  return position_v.xyz / position_v.w;
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

vec3 constructNormal(float depthA, vec2 texcoords, sampler2D depthtex) {
    const vec2 offsetB = vec2(0.0,0.001);
    const vec2 offsetC = vec2(0.001,0.0);
  
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
vec4 sample_biquadratic_exact(sampler2D channel, vec2 uv) {
    vec2 res = (textureSize(channel,0).xy);
    vec2 q = fract(uv * res);
    ivec2 t = ivec2(uv * res);
    const ivec3 e = ivec3(-1, 0, 1);
    vec4 s00 = texelFetch(channel, t + e.xx, 0);
    vec4 s01 = texelFetch(channel, t + e.xy, 0);
    vec4 s02 = texelFetch(channel, t + e.xz, 0);
    vec4 s12 = texelFetch(channel, t + e.yz, 0);
    vec4 s11 = texelFetch(channel, t + e.yy, 0);
    vec4 s10 = texelFetch(channel, t + e.yx, 0);
    vec4 s20 = texelFetch(channel, t + e.zx, 0);
    vec4 s21 = texelFetch(channel, t + e.zy, 0);
    vec4 s22 = texelFetch(channel, t + e.zz, 0);    
    vec2 q0 = (q+1.0)/2.0;
    vec2 q1 = q/2.0;	
    vec4 x0 = mix(mix(s00, s01, q0.y), mix(s01, s02, q1.y), q.y);
    vec4 x1 = mix(mix(s10, s11, q0.y), mix(s11, s12, q1.y), q.y);
    vec4 x2 = mix(mix(s20, s21, q0.y), mix(s21, s22, q1.y), q.y);    
	return mix(mix(x0, x1, q0.x), mix(x1, x2, q1.x), q.x);
}

void main() {


    vec3 reflection = vec3(1.0);
    

    vec4 color = texture(TranslucentSampler, texCoord);


    vec4 color2 = color;

    if (color.a > 0.01 ) {




    float deptha = texture(TranslucentDepthSampler, texCoord).r;

    vec3 normal = constructNormal(deptha, texCoord,  TranslucentDepthSampler);


////////////////////
    vec3 fragpos3 = toScreenSpace(vec3(texCoord-vec2(0)*oneTexel*0.5,deptha));



		float roughness = 0.1;

		float emissive = 0.0;
		float F0 = 0.02;

		vec3 reflectedVector = reflect(normalize(fragpos3), normal);
		float normalDotEye = dot(normal, normalize(fragpos3));
		float fresnel2 = pow(clamp(1.0 + normalDotEye,0.0,1.0), 5.0)*0.87+0.04;
		fresnel2 = mix(F0,1.0,fresnel2);

		

    vec4 screenPos = gl_FragCoord;
         screenPos.xy = (screenPos.xy / ScreenSize - vec2(0.5)) * 2.0;
         screenPos.zw = vec2(1.0);
    vec3 view = normalize((wgbufferModelViewInverse * screenPos).xyz);
    vec3 view2 = view;
         view2.y = -view2.y;

            vec3 suncol = texelFetch(temporals3Sampler,ivec2(8,37),0).rgb*0.5;

    vec3 sky_c = (mix(skyLut(view2,sunDir.xyz,view2.y,temporals3Sampler),suncol,0.5))  ;




		vec4 reflection = vec4(sky_c.rgb,0.);

		vec3 rtPos = rayTrace(reflectedVector,fragpos3.xyz,R2_dither(), fresnel2);
		if (rtPos.z <1.){
	    vec4 fragpositionPrev = gbufferProjectionInverse * vec4(rtPos*2.-1.,1.);
		fragpositionPrev /= fragpositionPrev.w;
	    reflection.a = 1.0;
		reflection.rgb = texture(TerrainCloudsSampler,rtPos.xy).rgb;
		}


		reflection.rgb = mix(sky_c.rgb, reflection.rgb, reflection.a);

        vec3 reflected= reflection.rgb*fresnel2;

        float alpha0 = color2.a;
	    color.a = -color2.a*fresnel2+color2.a+fresnel2;
		color.rgb =clamp((color2.rgb*6.5)/color.a*alpha0*(1.0-fresnel2)*0.1+(reflected*10)/color.a*0.1,0.0,1.0);
    //    color.rgb = vec3(normal.rgb);

    }        
   

    fragColor= vec4(color.rgba);
}
