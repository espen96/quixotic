#version 150

in vec2 texCoord;

uniform vec2 OutSize;
uniform sampler2D DiffuseSampler;
uniform sampler2D CurrentFrameDepthSampler;
uniform vec2 ScreenSize;
uniform sampler2D PreviousFrameSampler;
uniform sampler2D PreviousFrameDepthSampler;
uniform float Time;

out vec4 fragColor;

float pt = texelFetch(PreviousFrameSampler,ivec2(17,0),0).r;
#define VXAA_TEXTURE_CURRENT DiffuseSampler
#define VXAA_TEXTURE_PREV PreviousFrameSampler

#define VXAA_TEMPORALEDGE_THRES 0.05
#define VXAA_TEMPORALEDGE_TIME_MIN 0.0000001
#define VXAA_TEMPORALEDGE_TIME_MAX 1.15
#define VXAA_SPATIAL_FLICKER_TIME 2.35
#define VXAA_MORPHOLOGICAL_STRENGTH 0.85
#define VXAA_MORPHOLOGICAL_SHARPEN 0.05
#define iTimeDelta 1000.0/abs(Time - pt)
#define VXAA_W 0
#define VXAA_E 1
#define VXAA_N 2
#define VXAA_S 3
#define VXAA_NW 0
#define VXAA_NE 1
#define VXAA_SW 2
#define VXAA_SE 3

float saturate( float x )
{
    return clamp( x, 0.0, 1.0 );
}

vec4 pow3( vec4 x, float y )
{
    return vec4( pow( x.x, y ), pow( x.y, y ), pow( x.z, y ), x.w );
}

float VXAALuma( vec3 c )
{
    return dot( c, vec3( 0.2126, 0.7152, 0.0722 ) );
}

float VXAALumaDiff( vec3 x, vec3 y )
{
    float l1 = dot( x, vec3( 0.2126, 0.7152, 0.0722 ) );
    float l2 = dot( y, vec3( 0.2126, 0.7152, 0.0722 ) );
    return abs( l1 - l2 );
}

float VXAATemporalContrast( float currentLuma, float historyLuma )
{
    float x = saturate( abs( historyLuma - currentLuma ) - VXAA_TEMPORALEDGE_THRES );
    float x2 = x * x, x3 = x2 * x;
    return saturate( 3.082671957671837 * x - 3.9384920634917364 * x2 + 1.8518518518516354 * x3 );
}

float VXAAMorphStrengthShaper( float x )
{
    return 1.3 * x - 0.3 * x * x;
}

float VXAASpatialContrast( vec2 spatialLumaMinMax )
{
    float spatialContrast = spatialLumaMinMax.y - spatialLumaMinMax.x;
    return mix( 0.0f, 1.0f, spatialContrast );
}

float VXAATemporalFilterAlpha( float fpsRcp, float convergenceTime )
{
    return exp( -fpsRcp / convergenceTime );
}

vec3 VXAAClampHistory( vec3 history, vec4 currN[4] )
{
    vec3 cmin = min( min( currN[0].rgb, currN[1].rgb ), min( currN[2].rgb, currN[3].rgb ) );
    vec3 cmax = max( min( currN[0].rgb, currN[1].rgb ), max( currN[2].rgb, currN[3].rgb ) );
    return vec3(
        clamp( history.r, cmin.r, cmax.r ),
        clamp( history.g, cmin.g, cmax.g ),
        clamp( history.b, cmin.b, cmax.b )
    );
}

vec4 VXAASharpen( vec4 history, vec4 histN[4] )
{
    vec4 nh = histN[VXAA_NW] + histN[VXAA_NE] + histN[VXAA_SW] + histN[VXAA_SE];
    return mix( history, history * 5.0f - nh, VXAA_MORPHOLOGICAL_SHARPEN );
}

vec4 VXAAMorphological( vec2 uv, vec4 current, vec4 currN[4], float strength )
{
    if ( strength < 0.1f ) return current;
    float lumaNW = currN[VXAA_NW].a, lumaNE = currN[VXAA_NE].a,
        lumaSW = currN[VXAA_SW].a, lumaSE = currN[VXAA_SE].a;
    lumaNE += 0.0025;
    float lumaMin = min( current.a, min( min( lumaNW, lumaNE ), min( lumaSW, lumaSE ) ) );
    float lumaMax = max( current.a, max( max( lumaNW, lumaNE ), max( lumaSW, lumaSE ) ) );
    
    vec2 dir;
    dir.x = ( lumaSW - lumaNE ) + ( lumaSE - lumaNW );
    dir.y = ( lumaSW - lumaNE ) - ( lumaSE - lumaNW );
    vec2 dirN = normalize( dir );
    
    vec4 n1 = texture( VXAA_TEXTURE_CURRENT, uv - dirN * strength / ScreenSize.xy );
    vec4 p1 = texture( VXAA_TEXTURE_CURRENT, uv + dirN * strength / ScreenSize.xy );
    return ( n1 + p1 ) * 0.5;
}

vec4 VXAAFilmic( vec2 uv, vec4 current, vec4 history, vec4 currN[4], vec4 histN[4] )
{
    // Temporal contrast weight.
    float temporalContrastWeight = VXAATemporalContrast( current.a, history.a );

    // Spatial contrast weight.
    vec2 spatialLumaMinMaxC = vec2(
        min( min( currN[0].a, currN[1].a ), min( currN[2].a, currN[3].a ) ),
        max( max( currN[0].a, currN[1].a ), max( currN[2].a, currN[3].a ) )
    );
    vec2 spatialLumaMinMaxH = vec2(
        min( min( histN[0].a, histN[1].a ), min( histN[2].a, histN[3].a ) ),
        max( max( histN[0].a, histN[1].a ), max( histN[2].a, histN[3].a ) )
    );
    float spatialContrastWeightC = VXAASpatialContrast( spatialLumaMinMaxC );
    float spatialContrastWeightH = VXAASpatialContrast( spatialLumaMinMaxH );
    float spatialContrastWeight = abs( spatialContrastWeightC - spatialContrastWeightH );
    
    // Evaluate convergence time from weights.
    float convergenceTime = mix( VXAA_TEMPORALEDGE_TIME_MIN, VXAA_TEMPORALEDGE_TIME_MAX, temporalContrastWeight );
    convergenceTime = mix( convergenceTime, VXAA_SPATIAL_FLICKER_TIME, spatialContrastWeight );
    float alpha = VXAATemporalFilterAlpha( iTimeDelta, convergenceTime );

    
    // Apply morpholigical AA filter and sharpen.
    float strength = VXAAMorphStrengthShaper( spatialContrastWeightC * 4.0 ) * VXAA_MORPHOLOGICAL_STRENGTH;
    current = VXAAMorphological( uv, current, currN, strength );
    current = VXAASharpen( current, currN );
    
    // Clamp history to neighbourhood, and apply filmic blend.
    history.rgb = VXAAClampHistory( history.rgb, currN );
    current = mix( current, history, alpha );
    return current;
}
vec2 VXAADifferentialBlendWeight( vec4 n[4] )
{
    float diffWE = VXAALumaDiff( n[ VXAA_W ].rgb, n[ VXAA_E ].rgb );
    float diffNS = VXAALumaDiff( n[ VXAA_N ].rgb, n[ VXAA_S ].rgb );
    return diffWE < diffNS ? vec2( 0.5, 0.0 ) : vec2( 0.0, 0.5 );
}

vec4 VXAADifferentialBlend( vec4 n[4], vec2 w )
{
    vec4 c = vec4( 0.0 );
    c += ( n[ VXAA_W ] + n[ VXAA_E ] ) * w.x;
    c += ( n[ VXAA_N ] + n[ VXAA_S ] ) * w.y;
    return c;
}

void VXAAUpsampleT4x( out vec4 vtex[4], vec4 current, vec4 history, vec4 currN[4], vec4 histN[4] )
{
    vec4 n1[4], n2[4];
    
    n1[VXAA_W] = currN[VXAA_W];
    n1[VXAA_E] = current;
    n1[VXAA_N] = history;
    n1[VXAA_S] = histN[VXAA_S];
    
    n2[VXAA_W] = history;
    n2[VXAA_E] = histN[VXAA_E];
    n2[VXAA_N] = currN[VXAA_N];
    n2[VXAA_S] = current;
    
    
    vec4 weights = vec4( VXAADifferentialBlendWeight( n1 ), VXAADifferentialBlendWeight( n2 ) );
    vtex[VXAA_NW] = history;
    vtex[VXAA_NE] = VXAADifferentialBlend( n2, weights.zw );
    vtex[VXAA_SW] = VXAADifferentialBlend( n1, weights.xy );
    vtex[VXAA_SE] = current;
}

void main() {

    vec2 iResolution =ScreenSize;
    vec2 uv = texCoord;
    
    // Sample scene and neighbourhood.
    
    vec4 current = clamp( vec4( texture( VXAA_TEXTURE_CURRENT, uv ).rgb, 1.0 ), vec4( 0.0f ), vec4( 1.0f ) );
    vec4 history = clamp( vec4( texture( VXAA_TEXTURE_PREV, uv ).rgb, 1.0 ), vec4( 0.0f ), vec4( 1.0f ) );
    current.a = VXAALuma( current.rgb ); history.a = VXAALuma( history.rgb );
    
    vec4 currN[4];
    currN[VXAA_NW] = clamp( texture( VXAA_TEXTURE_CURRENT, uv + 0.6f * vec2( -1.0f,  1.0f ) / iResolution.xy ), vec4( 0.0f ), vec4( 1.0f ) );
    currN[VXAA_NE] = clamp( texture( VXAA_TEXTURE_CURRENT, uv + 0.6f * vec2(  1.0f,  1.0f ) / iResolution.xy ), vec4( 0.0f ), vec4( 1.0f ) );
    currN[VXAA_SW] = clamp( texture( VXAA_TEXTURE_CURRENT, uv + 0.6f * vec2( -1.0f, -1.0f ) / iResolution.xy ), vec4( 0.0f ), vec4( 1.0f ) );
    currN[VXAA_SE] = clamp( texture( VXAA_TEXTURE_CURRENT, uv + 0.6f * vec2(  1.0f, -1.0f ) / iResolution.xy ), vec4( 0.0f ), vec4( 1.0f ) );
    currN[VXAA_NW].a = VXAALuma( currN[VXAA_NW].rgb );
    currN[VXAA_NE].a = VXAALuma( currN[VXAA_NE].rgb );
    currN[VXAA_SW].a = VXAALuma( currN[VXAA_SW].rgb );
    currN[VXAA_SE].a = VXAALuma( currN[VXAA_SE].rgb );
    
    vec4 histN[4];
    histN[VXAA_NW] = clamp( texture( VXAA_TEXTURE_PREV, uv + 0.6f * vec2( -1.0f,  1.0f ) / iResolution.xy ), vec4( 0.0f ), vec4( 1.0f ) );
    histN[VXAA_NE] = clamp( texture( VXAA_TEXTURE_PREV, uv + 0.6f * vec2(  1.0f,  1.0f ) / iResolution.xy ), vec4( 0.0f ), vec4( 1.0f ) );
    histN[VXAA_SW] = clamp( texture( VXAA_TEXTURE_PREV, uv + 0.6f * vec2( -1.0f, -1.0f ) / iResolution.xy ), vec4( 0.0f ), vec4( 1.0f ) );
    histN[VXAA_SE] = clamp( texture( VXAA_TEXTURE_PREV, uv + 0.6f * vec2(  1.0f, -1.0f ) / iResolution.xy ), vec4( 0.0f ), vec4( 1.0f ) );
    histN[VXAA_NW].a = VXAALuma( histN[VXAA_NW].rgb );
    histN[VXAA_NE].a = VXAALuma( histN[VXAA_NE].rgb );
    histN[VXAA_SW].a = VXAALuma( histN[VXAA_SW].rgb );
    histN[VXAA_SE].a = VXAALuma( histN[VXAA_SE].rgb );
    


    // Filmic pass.    
    fragColor = VXAAFilmic( uv, current, history, currN, histN );


//}

if (gl_FragCoord.x < 18. && gl_FragCoord.y < 1.){

     fragColor = vec4(Time);
    }
}