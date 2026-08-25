package cn.tetraploid.rtwind.sensor.ui

import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.navigation.NavDestination.Companion.hierarchy
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import cn.tetraploid.rtwind.sensor.ui.screens.DashboardScreen
import cn.tetraploid.rtwind.sensor.ui.screens.SettingsScreen

private sealed class Route(val path: String, val label: String) {
    data object Dashboard : Route("dashboard", "仪表盘")
    data object Settings : Route("settings", "设置")
}

@Composable
fun RtwindSensorApp() {
    val nav = rememberNavController()
    val backStack by nav.currentBackStackEntryAsState()
    val destination = backStack?.destination

    Scaffold(
        bottomBar = {
            NavigationBar {
                listOf(Route.Dashboard, Route.Settings).forEach { route ->
                    NavigationBarItem(
                        selected = destination?.hierarchy?.any { it.route == route.path } == true,
                        onClick = { nav.navigate(route.path) { launchSingleTop = true } },
                        icon = {
                            Icon(
                                when (route) {
                                    Route.Dashboard -> Icons.Default.Home
                                    Route.Settings -> Icons.Default.Settings
                                },
                                contentDescription = route.label,
                            )
                        },
                        label = { Text(route.label) },
                    )
                }
            }
        },
    ) { padding ->
        NavHost(
            navController = nav,
            startDestination = Route.Dashboard.path,
            modifier = Modifier.padding(padding),
        ) {
            composable(Route.Dashboard.path) { DashboardScreen() }
            composable(Route.Settings.path) { SettingsScreen() }
        }
    }
}
